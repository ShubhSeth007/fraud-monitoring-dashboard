"""
monitor_drift.py — Evidently AI drift engine with Prometheus push and Slack alerting.

Usage:
    python monitor_drift.py                         # full report + alert if needed
    python monitor_drift.py --no-alert              # skip Slack notification
    python monitor_drift.py --threshold 0.05        # custom KS p-value threshold

Exit codes:
    0 — no significant drift detected
    1 — drift detected (useful for CI/CD gate)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from scipy import stats

# ── Config (override via environment variables) ────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
PROMETHEUS_PUSHGATEWAY = os.getenv("PROMETHEUS_PUSHGATEWAY", "")
KS_PVALUE_THRESHOLD = float(os.getenv("KS_PVALUE_THRESHOLD", "0.05"))
REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "data_drift_report.html")
SUMMARY_PATH = "drift_summary.json"


class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, drifted_features: list, summary: dict):
        if not self.webhook_url:
            print("[notify] SLACK_WEBHOOK_URL not set — skipping notification.")
            return

        feature_lines = "\n".join(
            f"  • `{f['feature']}` — KS stat: *{f['ks_statistic']:.4f}*, p-value: *{f['p_value']:.4f}*"
            for f in drifted_features
        )
        payload = {
            "text": ":rotating_light: *ML Model Data Drift Alert*",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "🚨 Fraud Detection Model — Drift Detected"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*{len(drifted_features)} feature(s)* crossed the KS p-value threshold "
                            f"of `{KS_PVALUE_THRESHOLD}`.\n\n{feature_lines}"
                        ),
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Total features checked:*\n{summary['total_features']}"},
                        {"type": "mrkdwn", "text": f"*Drifted features:*\n{summary['drifted_count']}"},
                        {"type": "mrkdwn", "text": f"*Drift rate:*\n{summary['drift_rate']:.1%}"},
                        {"type": "mrkdwn", "text": f"*Detected at:*\n{summary['timestamp']}"},
                    ],
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Trigger Retraining"},
                            "style": "danger",
                            "url": "https://github.com/YOUR_USERNAME/fraud-monitoring-dashboard/actions",
                        }
                    ],
                },
            ],
        }

        resp = requests.post(self.webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("[notify] Slack alert sent successfully.")
        else:
            print(f"[notify] Slack alert failed: {resp.status_code} {resp.text}")


def push_to_prometheus(drift_results: list):
    """Push drift scores to a Prometheus Pushgateway (optional)."""
    if not PROMETHEUS_PUSHGATEWAY:
        return

    lines = []
    for r in drift_results:
        safe_name = r["feature"].replace("-", "_")
        lines.append(f'feature_drift_ks_statistic{{feature="{r["feature"]}"}} {r["ks_statistic"]}')
        lines.append(f'feature_drift_pvalue{{feature="{r["feature"]}"}} {r["p_value"]}')
        lines.append(f'feature_drifted{{feature="{r["feature"]}"}} {int(r["drifted"])}')

    payload = "\n".join(lines) + "\n"
    try:
        resp = requests.post(
            f"{PROMETHEUS_PUSHGATEWAY}/metrics/job/fraud_drift_monitor",
            data=payload,
            headers={"Content-Type": "text/plain"},
            timeout=5,
        )
        print(f"[prometheus] Push status: {resp.status_code}")
    except Exception as e:
        print(f"[prometheus] Push failed: {e}")


def run_drift_analysis(
    reference_path: str = "train_baseline.csv",
    current_path: str = "production_logs.csv",
    send_alert: bool = True,
    ks_threshold: float = KS_PVALUE_THRESHOLD,
) -> dict:
    print("── Loading datasets ───────────────────────────────────────────────")
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        print(f"[error] Missing: {reference_path} or {current_path}")
        sys.exit(1)

    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)

    shared_cols = [c for c in ref.columns if c in cur.columns]
    print(f"  Reference: {ref.shape[0]} rows | Current: {cur.shape[0]} rows | Features: {len(shared_cols)}")

    # ── Per-feature KS tests ──────────────────────────────────────────────────
    print("\n── Kolmogorov–Smirnov tests ───────────────────────────────────────")
    drift_results = []
    for col in shared_cols:
        ks_stat, p_value = stats.ks_2samp(ref[col].dropna(), cur[col].dropna())
        drifted = p_value < ks_threshold
        drift_results.append({
            "feature": col,
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "mean_shift": round(float(cur[col].mean() - ref[col].mean()), 6),
            "std_ratio": round(float(cur[col].std() / (ref[col].std() + 1e-9)), 6),
            "drifted": drifted,
        })
        flag = "⚠️  DRIFT" if drifted else "✓  OK"
        print(f"  {flag:<12} {col:<30} KS={ks_stat:.4f}  p={p_value:.4f}")

    drifted_features = [r for r in drift_results if r["drifted"]]
    drift_rate = len(drifted_features) / len(drift_results) if drift_results else 0.0

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_features": len(drift_results),
        "drifted_count": len(drifted_features),
        "drift_rate": round(drift_rate, 4),
        "ks_threshold": ks_threshold,
        "drift_detected": len(drifted_features) > 0,
        "feature_results": drift_results,
    }

    # ── Evidently full report ─────────────────────────────────────────────────
    print("\n── Building Evidently report ──────────────────────────────────────")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref[shared_cols], current_data=cur[shared_cols])
    report.save_html(REPORT_PATH)
    print(f"  Saved: {REPORT_PATH}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {SUMMARY_PATH}")

    # ── Push to Prometheus ────────────────────────────────────────────────────
    push_to_prometheus(drift_results)

    # ── Alert ─────────────────────────────────────────────────────────────────
    print("\n── Alerting ───────────────────────────────────────────────────────")
    if drifted_features and send_alert:
        notifier = SlackNotifier(SLACK_WEBHOOK_URL)
        notifier.send(drifted_features, summary)
    elif not drifted_features:
        print("  No drift detected — no alert sent.")
    else:
        print("  Alerts disabled via --no-alert flag.")

    print(f"\n── Summary ────────────────────────────────────────────────────────")
    print(f"  Features checked : {summary['total_features']}")
    print(f"  Drifted          : {summary['drifted_count']} ({drift_rate:.1%})")
    print(f"  Status           : {'🚨 DRIFT DETECTED' if summary['drift_detected'] else '✅ CLEAN'}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift analysis on production logs.")
    parser.add_argument("--reference", default="train_baseline.csv")
    parser.add_argument("--current", default="production_logs.csv")
    parser.add_argument("--no-alert", action="store_true")
    parser.add_argument("--threshold", type=float, default=KS_PVALUE_THRESHOLD)
    args = parser.parse_args()

    summary = run_drift_analysis(
        reference_path=args.reference,
        current_path=args.current,
        send_alert=not args.no_alert,
        ks_threshold=args.threshold,
    )

    sys.exit(1 if summary["drift_detected"] else 0)
