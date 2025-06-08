#!/usr/bin/env python3
"""
Train GradientBoostingRegressor model for the legacy reimbursement
challenge and save it as `model.pkl` in the project root.

This script can be run once locally.  At inference time `run.sh` will load
`model.pkl` and use the same feature transformation (see src/features.py).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.features import make_features, FEATURE_NAMES

DATA_FILE = "public_cases.json"
MODEL_FILE = "model.pkl"


def load_dataset(path: str = DATA_FILE):
    with open(path, "r") as f:
        raw = json.load(f)
    X = []
    y = []
    for row in raw:
        d = row["input"]["trip_duration_days"]
        m = row["input"]["miles_traveled"]
        r = row["input"]["total_receipts_amount"]
        X.append(make_features(d, m, r))
        y.append(row["expected_output"])
    return np.array(X), np.array(y)


def main():
    print("📥 Loading data …")
    X, y = load_dataset()
    print(f"  → {X.shape[0]} rows, {X.shape[1]} features")

    print("🧠 Training GradientBoostingRegressor …")
    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    exact_matches = np.mean(np.isclose(preds, y, atol=0.01))
    close_matches = np.mean(np.isclose(preds, y, atol=1.0))
    print(f"  → MAE: ${mae:.2f}")
    print(f"  → Exact matches (±$0.01): {exact_matches:.1%}")
    print(f"  → Close matches  (±$1.00): {close_matches:.1%}")

    print(f"💾 Saving model to {MODEL_FILE}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print("✅ Done!")


if __name__ == "__main__":
    main() 