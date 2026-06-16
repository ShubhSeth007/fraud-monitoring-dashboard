"""
retrain_pipeline.py — Automated retraining triggered by drift detection.

Designed to run inside a GitHub Actions workflow when monitor_drift.py exits with code 1.
Steps:
  1. Pull latest production logs from configured data source (S3 / GCS / local)
  2. Merge with historical training baseline
  3. Retrain XGBoost with Optuna (fast: 15 trials)
  4. Validate new model beats old model on holdout PR-AUC
  5. Register new model in MLflow if validation passes
  6. Export fresh artefacts for Docker rebuild

Usage:
    python retrain_pipeline.py
    python retrain_pipeline.py --trials 20 --min-improvement 0.01
"""

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_NAME = "fraud-detector"
RETRAIN_LOG_PATH = "retrain_log.json"


def fetch_latest_data() -> pd.DataFrame:
    """
    Pull the most recent production logs.
    Extend this function with your actual data source:
      - AWS S3:  boto3.client('s3').download_file(...)
      - GCS:     storage.Client().bucket(...).blob(...).download_to_filename(...)
      - Local:   just read the CSV directly
    """
    path = os.getenv("PRODUCTION_DATA_PATH", "production_logs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Production data not found at {path}")
    df = pd.read_csv(path)
    print(f"[data] Loaded {len(df)} production records from {path}")
    return df


def merge_datasets(baseline_path: str = "train_baseline.csv") -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    production = fetch_latest_data()
    combined = pd.concat([baseline, production], ignore_index=True).drop_duplicates()
    print(f"[data] Combined dataset: {len(combined)} rows")
    return combined


def quick_objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 50, 200),
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "tree_method": "hist",
        "random_state": 42,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        m = XGBClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx], verbose=False)
        scores.append(average_precision_score(y[val_idx], m.predict_proba(X[val_idx])[:, 1]))
    return float(np.mean(scores))


def load_current_model_score(X_holdout, y_holdout) -> float:
    """Evaluate the currently deployed model on the holdout set."""
    try:
        model = joblib.load("fraud_model.joblib")
        proba = model.predict_proba(X_holdout)[:, 1]
        score = average_precision_score(y_holdout, proba)
        print(f"[baseline] Current model PR-AUC on holdout: {score:.4f}")
        return score
    except Exception as e:
        print(f"[baseline] Could not load current model: {e}. Assuming 0.0.")
        return 0.0


def retrain(n_trials: int = 15, min_improvement: float = 0.005):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection-retrain")

    df = merge_datasets()
    feature_cols = joblib.load("feature_columns.joblib") if os.path.exists("feature_columns.joblib") else [
        c for c in df.columns if c not in ("isFraud", "TransactionID")
    ]
    X = df[feature_cols].values
    y = df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df))

    # Hold out 15% for final validation
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    baseline_score = load_current_model_score(X_hold, y_hold)

    print(f"\n[optuna] Running {n_trials} retraining trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: quick_objective(t, X_train, y_train), n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"[optuna] Best CV PR-AUC: {study.best_value:.4f}")

    # Train final model with best params
    new_model = XGBClassifier(
        **best_params,
        eval_metric="aucpr",
        use_label_encoder=False,
        tree_method="hist",
        random_state=42,
    )
    new_model.fit(X_train, y_train)
    new_score = average_precision_score(y_hold, new_model.predict_proba(X_hold)[:, 1])
    improvement = new_score - baseline_score

    print(f"\n[validate] New model PR-AUC   : {new_score:.4f}")
    print(f"[validate] Baseline PR-AUC    : {baseline_score:.4f}")
    print(f"[validate] Improvement        : {improvement:+.4f}")

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_pr_auc": round(baseline_score, 6),
        "new_model_pr_auc": round(new_score, 6),
        "improvement": round(improvement, 6),
        "n_trials": n_trials,
        "deployed": False,
        "reason": "",
    }

    if improvement >= min_improvement:
        print(f"\n[deploy] Improvement {improvement:+.4f} ≥ threshold {min_improvement} — deploying.")

        with mlflow.start_run(run_name=f"retrain-auto-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"):
            mlflow.log_params(best_params)
            mlflow.log_metric("pr_auc_holdout", new_score)
            mlflow.log_metric("pr_auc_improvement", improvement)

            joblib.dump(new_model, "fraud_model.joblib")
            joblib.dump(feature_cols, "feature_columns.joblib")
            mlflow.log_artifact("fraud_model.joblib")

            mlflow.xgboost.log_model(
                new_model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

        # Promote to Production
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"[deploy] Model v{latest.version} promoted to Production.")
        log_entry["deployed"] = True

    else:
        log_entry["reason"] = f"Improvement {improvement:+.4f} below threshold {min_improvement}"
        print(f"\n[deploy] Skipping deployment — {log_entry['reason']}")

    with open(RETRAIN_LOG_PATH, "w") as f:
        json.dump(log_entry, f, indent=2)
    print(f"[log] Retrain summary saved: {RETRAIN_LOG_PATH}")

    return log_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    args = parser.parse_args()

    result = retrain(n_trials=args.trials, min_improvement=args.min_improvement)
    sys.exit(0 if result["deployed"] else 2)
