"""
generate_data.py — Synthetic dataset generator with configurable drift patterns.

Usage:
    python generate_data.py                             # default moderate drift
    python generate_data.py --drift-level severe        # heavy drift for demo
    python generate_data.py --drift-level none          # clean baseline clone
    python generate_data.py --rows 5000                 # custom dataset size
"""

import argparse
import numpy as np
import pandas as pd

FEATURES = [
    "TransactionAmt", "card1", "dist1", "TransactionHour",
    "time_delta_card1", "TransactionAmt_to_mean_card1",
    "TransactionAmt_to_std_card1", "C1", "C2",
]

DRIFT_PROFILES = {
    "none": {
        "TransactionAmt": {"mean_shift": 0.0, "std_scale": 1.0},
        "time_delta_card1": {"mean_shift": 0.0, "std_scale": 1.0},
    },
    "mild": {
        "TransactionAmt": {"mean_shift": 0.3, "std_scale": 1.1},
        "time_delta_card1": {"mean_shift": 0.2, "std_scale": 1.2},
    },
    "moderate": {
        "TransactionAmt": {"mean_shift": 0.75, "std_scale": 1.4},
        "time_delta_card1": {"mean_shift": 0.5, "std_scale": 2.1},
        "C1": {"mean_shift": 0.4, "std_scale": 1.3},
    },
    "severe": {
        "TransactionAmt": {"mean_shift": 1.5, "std_scale": 2.5},
        "time_delta_card1": {"mean_shift": 1.2, "std_scale": 3.0},
        "C1": {"mean_shift": 0.9, "std_scale": 2.0},
        "dist1": {"mean_shift": 0.7, "std_scale": 1.8},
        "TransactionHour": {"mean_shift": 0.5, "std_scale": 1.5},
    },
}


def generate_datasets(n_rows: int = 2000, drift_level: str = "moderate", seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    profile = DRIFT_PROFILES.get(drift_level, DRIFT_PROFILES["moderate"])

    print(f"── Generating reference dataset ({n_rows} rows) ───────────────────")
    ref_data = rng.standard_normal((n_rows, len(FEATURES)))
    df_ref = pd.DataFrame(ref_data, columns=FEATURES)

    # Realistic scaling
    df_ref["TransactionAmt"] = np.abs(df_ref["TransactionAmt"]) * 150 + 50
    df_ref["card1"] = np.abs(df_ref["card1"]) * 5000 + 1000
    df_ref["dist1"] = np.abs(df_ref["dist1"]) * 100
    df_ref["TransactionHour"] = (np.abs(df_ref["TransactionHour"]) * 4 + 12).clip(0, 23).astype(int)
    df_ref["time_delta_card1"] = np.abs(df_ref["time_delta_card1"]) * 300 + 60
    df_ref.to_csv("train_baseline.csv", index=False)
    print(f"  Saved: train_baseline.csv")

    print(f"\n── Generating production dataset (drift_level='{drift_level}') ──")
    cur_data = rng.standard_normal((n_rows, len(FEATURES)))
    df_cur = pd.DataFrame(cur_data, columns=FEATURES)

    # Same realistic scaling as baseline
    df_cur["TransactionAmt"] = np.abs(df_cur["TransactionAmt"]) * 150 + 50
    df_cur["card1"] = np.abs(df_cur["card1"]) * 5000 + 1000
    df_cur["dist1"] = np.abs(df_cur["dist1"]) * 100
    df_cur["TransactionHour"] = (np.abs(df_cur["TransactionHour"]) * 4 + 12).clip(0, 23).astype(int)
    df_cur["time_delta_card1"] = np.abs(df_cur["time_delta_card1"]) * 300 + 60

    # Apply configurable drift
    print("\n  Drift injections:")
    for feature, drift in profile.items():
        if feature in df_cur.columns:
            before_mean = df_cur[feature].mean()
            df_cur[feature] = (
                df_cur[feature] * drift["std_scale"] + drift["mean_shift"] * df_cur[feature].std()
            )
            after_mean = df_cur[feature].mean()
            print(f"    {feature:<35} mean: {before_mean:.2f} → {after_mean:.2f}  (std_scale={drift['std_scale']})")

    df_cur.to_csv("production_logs.csv", index=False)
    print(f"\n  Saved: production_logs.csv")

    # Quick sanity stats
    print(f"\n── Dataset summary ────────────────────────────────────────────────")
    print(f"{'Feature':<35} {'Ref mean':>10} {'Prod mean':>10} {'Shift':>10}")
    print("─" * 70)
    for col in FEATURES:
        ref_m = df_ref[col].mean()
        cur_m = df_cur[col].mean()
        shift = cur_m - ref_m
        flag = " ◀ drifted" if abs(shift) > 0.5 * df_ref[col].std() else ""
        print(f"{col:<35} {ref_m:>10.2f} {cur_m:>10.2f} {shift:>+10.2f}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument(
        "--drift-level",
        choices=["none", "mild", "moderate", "severe"],
        default="moderate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_datasets(n_rows=args.rows, drift_level=args.drift_level, seed=args.seed)
