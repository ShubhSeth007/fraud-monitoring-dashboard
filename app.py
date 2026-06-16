import os
import json
import hashlib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import shap
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
import logging

# ── Optional Redis (degrades gracefully if not available) ──────────────────────
try:
    import redis
    _redis = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _redis.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    logging.warning("Redis unavailable — deduplication disabled.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection MLOps Platform",
    description="Production fraud scoring with explainability, drift monitoring, and live metrics.",
    version="2.0.0",
)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["outcome"],          # APPROVED | FLAGGED_FOR_REVIEW | DUPLICATE
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
FRAUD_PROBABILITY_GAUGE = Gauge(
    "last_fraud_probability",
    "Fraud probability of the most recent scored transaction",
)
DRIFT_SCORE_GAUGE = Gauge(
    "feature_drift_score",
    "Wasserstein drift score per feature",
    ["feature"],
)

# ── Model artefacts ────────────────────────────────────────────────────────────
MODEL = None
FEATURE_COLUMNS = None
EXPLAINER = None
REPORT_PATH = "data_drift_report.html"
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


def load_model():
    global MODEL, FEATURE_COLUMNS, EXPLAINER
    MODEL = joblib.load("fraud_model.joblib")
    FEATURE_COLUMNS = joblib.load("feature_columns.joblib")
    EXPLAINER = shap.TreeExplainer(MODEL)
    log.info("Model, feature schema, and SHAP explainer loaded.")


def build_drift_report():
    if not os.path.exists("train_baseline.csv") or not os.path.exists("production_logs.csv"):
        log.warning("Monitoring CSVs not found — skipping drift report.")
        return

    reference = pd.read_csv("train_baseline.csv")
    current = pd.read_csv("production_logs.csv")

    # Feature-level drift scores written to Prometheus gauges
    for col in reference.columns:
        if col in current.columns:
            ks_stat = float(np.abs(reference[col].mean() - current[col].mean()))
            DRIFT_SCORE_GAUGE.labels(feature=col).set(ks_stat)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(REPORT_PATH)
    log.info("Evidently drift report compiled.")


@app.on_event("startup")
def startup():
    load_model()
    build_drift_report()


# ── Request / response schemas ─────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Idempotency key for deduplication")
    TransactionAmt: float
    card1: float
    card2: float
    addr1: float
    dist1: float
    C1: float; C2: float; C3: float; C4: float; C5: float
    C6: float; C7: float; C8: float; C9: float; C10: float
    TransactionHour: int
    TransactionAmt_to_mean_card1: float
    TransactionAmt_to_std_card1: float
    time_delta_card1: float


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="New fraud decision threshold (0–1)")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _payload_hash(payload: dict) -> str:
    """Stable hash of the feature payload for deduplication."""
    canonical = json.dumps({k: v for k, v in payload.items() if k != "transaction_id"}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _check_cache(key: str):
    if not REDIS_AVAILABLE:
        return None
    cached = _redis.get(f"pred:{key}")
    return json.loads(cached) if cached else None


def _write_cache(key: str, result: dict, ttl: int = 60):
    if REDIS_AVAILABLE:
        _redis.setex(f"pred:{key}", ttl, json.dumps(result))


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "operational",
        "model_loaded": MODEL is not None,
        "redis_connected": REDIS_AVAILABLE,
        "drift_report_ready": os.path.exists(REPORT_PATH),
        "fraud_threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(tx: TransactionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    payload = tx.dict()
    cache_key = tx.transaction_id or _payload_hash(payload)

    # ── Deduplication ──────────────────────────────────────────────────────────
    cached = _check_cache(cache_key)
    if cached:
        PREDICTIONS_TOTAL.labels(outcome="DUPLICATE").inc()
        return {**cached, "cached": True}

    start = time.perf_counter()
    features = {k: payload[k] for k in FEATURE_COLUMNS}
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    prob = float(MODEL.predict_proba(X)[0][1])
    latency = time.perf_counter() - start

    is_fraud = int(prob >= THRESHOLD)
    status = "FLAGGED_FOR_REVIEW" if is_fraud else "APPROVED"

    # ── Metrics ────────────────────────────────────────────────────────────────
    PREDICTION_LATENCY.observe(latency)
    FRAUD_PROBABILITY_GAUGE.set(prob)
    PREDICTIONS_TOTAL.labels(outcome=status).inc()

    result = {
        "transaction_id": cache_key,
        "fraud_probability": round(prob, 4),
        "is_fraud": is_fraud,
        "status": status,
        "threshold_used": THRESHOLD,
        "latency_ms": round(latency * 1000, 2),
        "cached": False,
    }
    _write_cache(cache_key, result)
    return result


@app.post("/explain")
def explain(tx: TransactionRequest):
    """Returns SHAP feature-importance values for a single transaction."""
    if MODEL is None or EXPLAINER is None:
        raise HTTPException(status_code=503, detail="Model or explainer not loaded.")

    payload = tx.dict()
    features = {k: payload[k] for k in FEATURE_COLUMNS}
    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    shap_values = EXPLAINER.shap_values(X)
    # For binary classifiers shap_values is a list [class0, class1]
    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    explanation = sorted(
        [{"feature": f, "shap_value": round(float(v), 6)} for f, v in zip(FEATURE_COLUMNS, values)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )
    prob = float(MODEL.predict_proba(X)[0][1])
    return {
        "fraud_probability": round(prob, 4),
        "base_value": round(float(EXPLAINER.expected_value[1] if isinstance(EXPLAINER.expected_value, (list, np.ndarray)) else EXPLAINER.expected_value), 6),
        "top_features": explanation[:10],
    }


@app.post("/threshold")
def update_threshold(body: ThresholdUpdate):
    """Hot-update the fraud decision threshold without redeployment."""
    global THRESHOLD
    old = THRESHOLD
    THRESHOLD = body.threshold
    log.info(f"Threshold updated: {old} → {THRESHOLD}")
    return {"previous_threshold": old, "new_threshold": THRESHOLD}


@app.post("/refresh-drift")
def refresh_drift(background_tasks: BackgroundTasks):
    """Trigger a background Evidently report rebuild (non-blocking)."""
    background_tasks.add_task(build_drift_report)
    return {"message": "Drift report rebuild scheduled."}


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(status_code=500, detail="Drift report not yet generated. Call /refresh-drift.")
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
