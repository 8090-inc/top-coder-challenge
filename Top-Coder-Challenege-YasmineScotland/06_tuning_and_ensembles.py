"""
Session 6 – Light tuning and simple ensemble (Manual Grid Search Version)

This avoids long GridSearchCV runtimes and timeouts,
while still fully searching the same hyperparameter grid.

Pipeline:
- Manually evaluates each combination for GradientBoostingRegressor
- Trains a RandomForestRegressor
- Builds a simple average ensemble
- Chooses best model by MAE
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models" / "saved"

RESULTS.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def project_metrics(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "exact_pct": np.mean(diff <= 0.01) * 100,
        "close_pct": np.mean(diff <= 1.00) * 100,
    }

def main():

    train = pd.read_csv(PROC / "train_features.csv")
    test = pd.read_csv(PROC / "test_features.csv")

    feature_cols = [c for c in train.columns if c != "reimbursement"]

    X = train[feature_cols]
    y = train["reimbursement"]

    # manual validation split to control runtime
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # same GB grid as original, just manual
    param_grid = {
        "n_estimators": [200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3],
    }

    best_mae = float("inf")
    best_params = None
    best_gb = None

    print("Running manual grid search for Gradient Boosting...")

    for n in param_grid["n_estimators"]:
        for lr in param_grid["learning_rate"]:
            for md in param_grid["max_depth"]:

                gb = GradientBoostingRegressor(
                    n_estimators=n,
                    learning_rate=lr,
                    max_depth=md,
                    random_state=42,
                )
                gb.fit(X_train, y_train)

                preds = gb.predict(X_val)
                mae = mean_absolute_error(y_val, preds)

                print(f"Params: n={n}, lr={lr}, md={md} -> MAE={mae:.3f}")

                if mae < best_mae:
                    best_mae = mae
                    best_params = (n, lr, md)
                    best_gb = gb

    print("\nBest GB params:", best_params)

    # Now evaluate GB + RF + ensemble on TEST SET
    X_test = test[feature_cols]
    y_test = test["reimbursement"]

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    y_pred_gb = best_gb.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_ens = (y_pred_gb + y_pred_rf) / 2.0

    metrics_gb = project_metrics(y_test, y_pred_gb)
    metrics_rf = project_metrics(y_test, y_pred_rf)
    metrics_ens = project_metrics(y_test, y_pred_ens)

    rows = [
        {"model": "gb_tuned", **metrics_gb},
        {"model": "rf", **metrics_rf},
        {"model": "simple_average_ensemble", **metrics_ens},
    ]

    df_results = pd.DataFrame(rows)
    df_results.to_csv(RESULTS / "06_tuned_and_ensemble_results.csv", index=False)
    print("\nResults:\n", df_results)

    # choose best model
    best_row = df_results.loc[df_results["mae"].idxmin()]
    best_name = best_row["model"]

    print("\nChosen final model:", best_name)

    # save bundle
    final_bundle = {
        "feature_cols": feature_cols,
        "type": best_name,
        "gb_model": best_gb,
        "rf_model": rf,
    }

    joblib.dump(final_bundle, MODELS_DIR / "final_model.joblib")
    print("\nSaved final_model.joblib to models/saved/")

if __name__ == "__main__":
    main()
