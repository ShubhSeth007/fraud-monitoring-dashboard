"""
train.py — XGBoost + Optuna + MLflow | Designed for Kaggle IEEE-CIS Fraud Detection dataset.

Kaggle dataset: https://www.kaggle.com/competitions/ieee-fraud-detection/data
Required files in /kaggle/input/ieee-fraud-detection/:
    - train_transaction.csv
    - train_identity.csv  (optional, used if present)

Usage in Kaggle notebook:
    !python train.py
    !python train.py --trials 50
    !python train.py --trials 30 --register
"""

import argparse
import warnings
import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME    = "fraud-detection"
MODEL_NAME         = "fraud-detector"

# ── Kaggle IEEE-CIS dataset paths ──────────────────────────────────────────────
TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
IDENTITY_PATH    = "/kaggle/input/ieee-fraud-detection/train_identity.csv"


def load_and_engineer(transaction_path: str, identity_path: str) -> tuple:
    print("[data] Loading train_transaction.csv ...")
    df = pd.read_csv(transaction_path)
    print(f"[data] Raw shape: {df.shape} | Fraud rate: {df['isFraud'].mean():.4%}")

    # ── Optional: merge identity table ────────────────────────────────────────
    if os.path.exists(identity_path):
        print("[data] Merging train_identity.csv ...")
        identity = pd.read_csv(identity_path)
        df = df.merge(identity, on="TransactionID", how="left")
        print(f"[data] After merge: {df.shape}")

    # ── Feature engineering ────────────────────────────────────────────────────
    print("[data] Engineering features ...")

    # Time features
    df["TransactionHour"] = (df["TransactionDT"] // 3600) % 24
    df["TransactionDay"]  = (df["TransactionDT"] // 86400) % 7

    # Card-level aggregations (behavioural features)
    df["TransactionAmt_to_mean_card1"] = df["TransactionAmt"] / (
        df.groupby("card1")["TransactionAmt"].transform("mean") + 1e-9
    )
    df["TransactionAmt_to_std_card1"] = df["TransactionAmt"] / (
        df.groupby("card1")["TransactionAmt"].transform("std") + 1e-9
    )

    # Time delta between transactions on same card
    df = df.sort_values("TransactionDT")
    df["time_delta_card1"] = df.groupby("card1")["TransactionDT"].diff().fillna(0)

    # ── Select final feature set ───────────────────────────────────────────────
    # Core numeric features that exist in the IEEE-CIS dataset
    base_features = [
        "TransactionAmt",
        "card1", "card2",
        "addr1", "dist1",
        "C1", "C2", "C3", "C4", "C5",
        "C6", "C7", "C8", "C9", "C10",
        "TransactionHour", "TransactionDay",
        "TransactionAmt_to_mean_card1",
        "TransactionAmt_to_std_card1",
        "time_delta_card1",
    ]

    # Only keep columns that actually exist in this dataset version
    feature_cols = [c for c in base_features if c in df.columns]
    print(f"[data] Using {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].fillna(-999).values
    y = df["isFraud"].values

    print(f"[data] Final — rows: {len(y)} | fraud: {y.sum()} ({y.mean():.4%}) | features: {len(feature_cols)}")
    return X, y, feature_cols


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 200, 800),
        "max_depth"        : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
        "gamma"            : trial.suggest_float("gamma", 0, 3),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 50, 300),
        "eval_metric"      : "aucpr",
        "use_label_encoder": False,
        "tree_method"      : "hist",
        "device"           : "cuda",   # Kaggle GPU — falls back to CPU if unavailable
        "random_state"     : 42,
        "n_jobs"           : -1,
    }

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = XGBClassifier(**params)
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        prob = model.predict_proba(X[val_idx])[:, 1]
        score = average_precision_score(y[val_idx], prob)
        pr_aucs.append(score)
        print(f"  fold {fold+1}/5  PR-AUC={score:.4f}")

    mean_score = float(np.mean(pr_aucs))
    print(f"  → trial mean PR-AUC: {mean_score:.4f}")
    return mean_score


def train(n_trials: int = 30, register: bool = False):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, feature_cols = load_and_engineer(TRANSACTION_PATH, IDENTITY_PATH)

    print(f"\n[train] Running Optuna ({n_trials} trials) — optimising PR-AUC ...")
    study = optuna.create_study(direction="maximize", study_name="pr_auc")
    study.optimize(
        lambda t: objective(t, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params  = study.best_params
    best_pr_auc  = study.best_value
    print(f"\n[train] ✓ Best PR-AUC : {best_pr_auc:.4f}")
    print(f"[train] ✓ Best params : {best_params}")

    with mlflow.start_run(run_name=f"xgboost-optuna-{n_trials}trials") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("best_pr_auc_cv",  best_pr_auc)
        mlflow.log_metric("n_optuna_trials",  n_trials)
        mlflow.log_metric("fraud_rate",       float(y.mean()))
        mlflow.log_metric("n_features",       len(feature_cols))
        mlflow.log_metric("n_rows",           len(y))

        print("\n[train] Training final model on full dataset ...")
        final_params = {**best_params, "eval_metric": "aucpr",
                        "use_label_encoder": False, "tree_method": "hist",
                        "device": "cuda", "random_state": 42, "n_jobs": -1}
        final_model = XGBClassifier(**final_params)
        final_model.fit(X, y)

        # Save artefacts
        joblib.dump(final_model,  "/kaggle/working/fraud_model.joblib")
        joblib.dump(feature_cols, "/kaggle/working/feature_columns.joblib")
        print("[train] ✓ Saved: fraud_model.joblib")
        print("[train] ✓ Saved: feature_columns.joblib")

        mlflow.log_artifact("/kaggle/working/fraud_model.joblib")
        mlflow.log_artifact("/kaggle/working/feature_columns.joblib")

        mlflow.xgboost.log_model(
            final_model,
            name="model",
            registered_model_name=MODEL_NAME if register else None,
        )
        print(f"[train] MLflow run: {run.info.run_id}")

    if register:
        client   = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest   = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=MODEL_NAME, version=latest.version,
            stage="Production", archive_existing_versions=True,
        )
        print(f"[train] Model v{latest.version} → Production registry.")

    print("\n[train] ✓ Complete. Download from /kaggle/working/")
    print("         fraud_model.joblib")
    print("         feature_columns.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",   type=int,  default=30)
    parser.add_argument("--register", action="store_true")
    args, _ = parser.parse_known_args()
    train(n_trials=args.trials, register=args.register)
