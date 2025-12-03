"""
Session 4 – Baseline models.

Implements a couple of simple baselines:
- Mean predictor
- Plain linear regression on engineered features
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)


def project_metrics(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    exact = np.mean(diff <= 0.01) * 100
    close = np.mean(diff <= 1.00) * 100
    return {
        "mae": mae,
        "rmse": rmse,
        "exact_pct": exact,
        "close_pct": close,
    }


def main() -> None:
    train_path = PROC / "train_features.csv"
    test_path = PROC / "test_features.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run 03_feature_engineering.py first to create feature CSVs")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature_cols = [c for c in train.columns if c != "reimbursement"]

    X_train = train[feature_cols]
    y_train = train["reimbursement"]
    X_test = test[feature_cols]
    y_test = test["reimbursement"]

    rows = []

    # Baseline 1 – mean
    mean_value = y_train.mean()
    y_pred_mean = np.full_like(y_test, fill_value=mean_value, dtype=float)
    rows.append({"model": "mean", **project_metrics(y_test, y_pred_mean)})

    # Baseline 2 – linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rows.append({"model": "linear_regression", **project_metrics(y_test, y_pred_lr)})

    df_results = pd.DataFrame(rows)
    df_results.to_csv(RESULTS / "04_baseline_results.csv", index=False)
    print(df_results)


if __name__ == "__main__":
    main()
