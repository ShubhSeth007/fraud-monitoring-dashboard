# Fraud Detection MLOps Platform

An end-to-end, production-grade machine learning system for real-time financial fraud detection. Built with an **XGBoost** classifier served via **FastAPI**, containerized with **Docker**, and deployed live on **Render** — with full MLOps observability via **Evidently AI** and **Prometheus**.

🔴 **Live:** [https://fraud-metrics-dashboard.onrender.com](https://fraud-metrics-dashboard.onrender.com)

---

## Architecture

```
IEEE-CIS Dataset (Kaggle)
        │
        ▼
   train.py ──── Optuna hyperparameter search (30 trials)
        │         XGBoost + StratifiedKFold + PR-AUC optimisation
        │         MLflow experiment tracking
        ▼
fraud_model.joblib
feature_columns.joblib
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
   app.py (FastAPI)                         monitor_drift.py
   /predict  — XGBoost scoring             Evidently AI drift report
   /explain  — SHAP feature importance     KS-test per feature
   /threshold — hot-swap decision cutoff   Slack alerting
   /dashboard — live Evidently report      Prometheus push
   /metrics  — Prometheus scrape
        │
        ▼
   Docker container → Render (live)
        │
        ▼
   GitHub Actions (retrain.yml)
   Triggered on drift → auto-retrains → promotes new model
```

---

## Live Endpoints

| Endpoint | Method | Description |
|---|---|---|
| [`/health`](https://fraud-metrics-dashboard.onrender.com/health) | GET | Service + model readiness check |
| [`/docs`](https://fraud-metrics-dashboard.onrender.com/docs) | GET | Interactive Swagger UI |
| [`/predict`](https://fraud-metrics-dashboard.onrender.com/docs#/default/predict_predict_post) | POST | Real-time fraud scoring |
| [`/explain`](https://fraud-metrics-dashboard.onrender.com/docs#/default/explain_explain_post) | POST | SHAP feature importance per transaction |
| [`/threshold`](https://fraud-metrics-dashboard.onrender.com/docs#/default/update_threshold_threshold_post) | POST | Hot-update decision threshold (no redeploy) |
| [`/dashboard`](https://fraud-metrics-dashboard.onrender.com/dashboard) | GET | Live Evidently AI drift report |
| [`/metrics`](https://fraud-metrics-dashboard.onrender.com/metrics) | GET | Prometheus scrape endpoint |
| [`/refresh-drift`](https://fraud-metrics-dashboard.onrender.com/docs#/default/refresh_drift_refresh_drift_post) | POST | Trigger background drift report rebuild |

---

## Key Technical Features

**Extreme class imbalance handling** — trained on the IEEE-CIS dataset (3.5% fraud rate) using stratified cross-validation and tuned `scale_pos_weight`. Optimises PR-AUC rather than accuracy, which is a meaningless metric on skewed data.

**Hyperparameter optimisation with Optuna** — 30-trial Bayesian search over the full XGBoost parameter space targeting PR-AUC across 5 stratified folds. Best trial promoted automatically.

**MLflow experiment tracking** — every training run logs parameters, metrics, and model artefacts. Best model registered to the MLflow model registry and promoted to Production stage.

**SHAP explainability** — the `/explain` endpoint returns per-feature SHAP values for any transaction, making every decision auditable. Loaded lazily at runtime to stay within the 512MB free-tier memory limit.

**Evidently AI drift monitoring** — compares live production feature distributions against the training baseline using the Kolmogorov–Smirnov test. Generates a full interactive HTML report served at `/dashboard`.

**Prometheus metrics** — exposes `fraud_predictions_total`, `prediction_latency_seconds`, `last_fraud_probability`, and `feature_drift_score` for Grafana dashboarding.

**Redis deduplication** — idempotent prediction API; identical transaction payloads return cached results within a 60-second TTL window (degrades gracefully when Redis is unavailable).

**Dynamic threshold** — fraud decision cutoff configurable at runtime via `POST /threshold` without redeployment.

**GitHub Actions retraining pipeline** — runs every 6 hours, executes drift analysis, and triggers automatic retraining if any feature's KS p-value drops below 0.05. New model deployed only if it beats the current model's PR-AUC by at least 0.5%.

---

## Model Performance

Standard accuracy is meaningless for fraud detection — a model that always predicts "not fraud" achieves 96.5% accuracy on the IEEE-CIS dataset while catching zero fraud cases.

This model targets **PR-AUC (Average Precision)**:

| Metric | Score |
|---|---|
| Baseline (random guessing) | 0.035 |
| Optimised model PR-AUC | **0.2638** |

The tuned model performs over **7× better than random chance** on the positive class, with `scale_pos_weight` tuned to aggressively suppress false negatives (missed fraud) while keeping false positives manageable.

---

## Repository Structure

```
├── app.py                    # FastAPI serving engine
├── train.py                  # XGBoost + Optuna + MLflow training script
├── monitor_drift.py          # Evidently AI + KS tests + Slack alerting
├── retrain_pipeline.py       # Automated retraining with model validation gate
├── generate_monitoring_data.py  # Extracts real feature distributions from IEEE-CIS
├── generate_data.py          # Synthetic data generator (configurable drift levels)
├── fraud_model.joblib        # Serialised XGBoost model (trained on Kaggle)
├── feature_columns.joblib    # Feature schema for inference alignment
├── train_baseline.csv        # Reference feature distributions (from training data)
├── production_logs.csv       # Production feature sample with injected drift
├── Dockerfile                # Single-worker container (memory-optimised for free tier)
├── requirements.txt          # Serving dependencies only (no training libs)
├── grafana/
│   └── dashboard.json        # Pre-built Grafana dashboard (import directly)
└── .github/
    └── workflows/
        └── retrain.yml       # CI/CD drift detection + auto-retraining pipeline
```

---

## Local Setup

### Run via Docker

```bash
git clone https://github.com/YOUR_USERNAME/fraud-monitoring-dashboard.git
cd fraud-monitoring-dashboard

docker build -t fraud-detection-service:v2 .
docker run -p 8000:10000 -e PORT=10000 fraud-detection-service:v2
```

Visit `http://localhost:8000/docs`

### Run without Docker

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage

### Score a transaction

```bash
curl -X POST https://fraud-metrics-dashboard.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 150.50,
    "card1": 1234.0,
    "card2": 567.0,
    "addr1": 89.0,
    "dist1": 12.5,
    "C1": 1.0, "C2": 2.0, "C3": 0.0, "C4": 0.0, "C5": 1.0,
    "C6": 1.0, "C7": 0.0, "C8": 0.0, "C9": 1.0, "C10": 0.0,
    "TransactionHour": 14,
    "TransactionAmt_to_mean_card1": 1.12,
    "TransactionAmt_to_std_card1": 0.45,
    "time_delta_card1": 360.0
  }'
```

Response:

```json
{
  "transaction_id": "a3f9c...",
  "fraud_probability": 0.0241,
  "is_fraud": 0,
  "status": "APPROVED",
  "threshold_used": 0.5,
  "latency_ms": 18.4,
  "cached": false
}
```

### Get SHAP explanation

```bash
curl -X POST https://fraud-metrics-dashboard.onrender.com/explain \
  -H "Content-Type: application/json" \
  -d '{ ...same payload as /predict... }'
```

Response:

```json
{
  "fraud_probability": 0.0241,
  "base_value": 0.034,
  "top_features": [
    {"feature": "TransactionAmt", "shap_value": -0.012},
    {"feature": "time_delta_card1", "shap_value": -0.009},
    ...
  ]
}
```

### Update decision threshold

```bash
curl -X POST https://fraud-metrics-dashboard.onrender.com/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.3}'
```

---

## MLOps Observability

### Run drift analysis locally

```bash
python monitor_drift.py
# exits with code 1 if drift detected — usable as a CI gate
```

### Trigger retraining manually

```bash
python retrain_pipeline.py --trials 20 --min-improvement 0.005
```

### Grafana dashboard

Import `grafana/dashboard.json` into any Grafana instance pointed at your Prometheus endpoint. Pre-built panels include prediction throughput, P50/P95/P99 latency, fraud flag rate, and per-feature drift scores.

---

## Deployment

The service is containerised and deployed on Render via Docker. Render detects the `Dockerfile` automatically on push.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `10000` | Port the uvicorn server binds to |
| `FRAUD_THRESHOLD` | `0.5` | Initial fraud decision cutoff |
| `SLACK_WEBHOOK_URL` | — | Slack webhook for drift alerts |
| `REDIS_HOST` | `localhost` | Redis host for deduplication |
| `PROMETHEUS_PUSHGATEWAY` | — | Pushgateway URL for drift metrics |
