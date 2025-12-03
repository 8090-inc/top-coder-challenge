"""
Session 7 – Production pipeline (NO dataclass version)

Loads the trained model bundle (final_model.joblib)
and exposes a simple reusable predictor class.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Adjust this path if needed
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "saved" / "final_model.joblib"


class ReimbursementPredictor:
    """Helper class to load the final bundled model and make predictions."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Run 06_tuning_and_ensembles.py first."
            )

        # Load model bundle
        bundle = joblib.load(self.model_path)

        self.feature_cols = bundle["feature_cols"]     # list of feature names
        self.type = bundle["type"]                     # 'gb_tuned' or 'rf' or 'simple_average_ensemble'
        self.gb_model = bundle["gb_model"]             # GradientBoosting model
        self.rf_model = bundle["rf_model"]             # RandomForest model

    def _make_feature_row(self, days, miles, receipts):
        """
        Recreate **exactly** the engineered features from Session 3.
        """
        days_safe = days if days > 0 else 1.0

        data = {
            "trip_duration_days": float(days),
            "miles_traveled": float(miles),
            "total_receipts_amount": float(receipts),

            # engineered features
            "receipts_per_day": receipts / days_safe,
            "miles_per_day": miles / days_safe,
            "log_receipts": np.log1p(receipts),
            "log_miles": np.log1p(miles),
            "is_week_plus": int(days >= 7),
            "is_long_miles": int(miles > 500),
        }

        df = pd.DataFrame([data])
        return df[self.feature_cols]   # ensures correct column order

    def predict_one(self, days, miles, receipts):
        """
        Make a single prediction and return a rounded reimbursement amount.
        """
        X = self._make_feature_row(days, miles, receipts)

        if self.type == "gb_tuned":
            pred = self.gb_model.predict(X)[0]

        elif self.type == "rf":
            pred = self.rf_model.predict(X)[0]

        else:
            # simple average ensemble
            p_gb = self.gb_model.predict(X)[0]
            p_rf = self.rf_model.predict(X)[0]
            pred = (p_gb + p_rf) / 2.0

        # Round to 2 decimals, clip negatives
        pred = max(0.0, round(float(pred), 2))
        return pred


# Small check to confirm everything works
def quick_self_test():
    predictor = ReimbursementPredictor()
    cases = [
        (1, 50, 80.0),
        (3, 300, 400.0),
        (7, 900, 1200.0),
    ]

    for d, m, r in cases:
        out = predictor.predict_one(d, m, r)
        print(f"Input: days={d}, miles={m}, receipts={r:.2f} -> {out:.2f}")


if __name__ == "__main__":
    quick_self_test()
