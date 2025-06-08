"""
V4 Ensemble-Corrected Reimbursement Model

This model combines:
1. V3 rule engine (cluster-based linear models) as base predictor
2. Ensemble of tree models (ExtraTrees + GBM + RF) for residual correction

Performance: MAE $24.09 on training data (79% improvement over rule engine alone)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Import v3 rule engine
try:
    from cluster_models_optimized import calculate_reimbursement_v3
except ImportError:
    from models.cluster_models_optimized import calculate_reimbursement_v3


class EnsembleCorrectedModel:
    """Ensemble-corrected reimbursement calculator"""
    
    def __init__(self, model_dir=None):
        """Initialize with trained ensemble models"""
        if model_dir is None:
            # Try to find the model directory relative to this file
            base_path = Path(__file__).parent.parent
            model_dir = base_path / 'analysis' / '08_ensemble_residual_corrector'
        self.model_dir = Path(model_dir)
        self.models = {}
        self.weights = {'ExtraTrees': 0.4, 'GBM': 0.35, 'RandomForest': 0.25}
        self._load_models()
    
    def _load_models(self):
        """Load the trained residual correction models"""
        for model_name in self.weights.keys():
            model_path = self.model_dir / f'{model_name.lower()}_residual_model.pkl'
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
            else:
                print(f"Warning: Model {model_path} not found")
    
    def engineer_features(self, trip_days, miles_traveled, total_receipts_amount, rule_engine_pred):
        """
        Engineer features for the ensemble models.
        Must match the feature engineering in training exactly.
        """
        # Create single-row DataFrame
        df = pd.DataFrame({
            'trip_days': [trip_days],
            'miles_traveled': [miles_traveled],
            'total_receipts_amount': [total_receipts_amount],
            'rule_engine_pred': [rule_engine_pred]
        })
        
        features = pd.DataFrame()
        
        # Basic features
        features['trip_days'] = df['trip_days']
        features['miles_traveled'] = df['miles_traveled']
        features['total_receipts_amount'] = df['total_receipts_amount']
        features['rule_engine_pred'] = df['rule_engine_pred']
        
        # Derived features
        features['miles_per_day'] = df['miles_traveled'] / df['trip_days']
        features['receipts_per_day'] = df['total_receipts_amount'] / df['trip_days']
        features['receipts_per_mile'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1)
        
        # Log transforms
        features['log_miles'] = np.log1p(df['miles_traveled'])
        features['log_receipts'] = np.log1p(df['total_receipts_amount'])
        features['log_days'] = np.log1p(df['trip_days'])
        
        # Polynomial features
        features['days_squared'] = df['trip_days'] ** 2
        features['miles_squared'] = df['miles_traveled'] ** 2
        features['receipts_squared'] = df['total_receipts_amount'] ** 2
        
        # Interaction features
        features['days_x_miles'] = df['trip_days'] * df['miles_traveled']
        features['days_x_receipts'] = df['trip_days'] * df['total_receipts_amount']
        features['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
        
        # Efficiency features
        features['is_efficient'] = ((df['miles_traveled'] / df['trip_days'] >= 180) & 
                                    (df['miles_traveled'] / df['trip_days'] <= 220)).astype(int)
        features['is_high_miles'] = (df['miles_traveled'] > 1000).astype(int)
        features['is_low_receipts'] = (df['total_receipts_amount'] < 50).astype(int)
        
        # Receipt ending features
        receipts_cents = (df['total_receipts_amount'] * 100).astype(int) % 100
        features['is_49_ending'] = (receipts_cents == 49).astype(int)
        features['is_99_ending'] = (receipts_cents == 99).astype(int)
        
        # Trip type indicators
        features['is_single_day'] = (df['trip_days'] == 1).astype(int)
        features['is_long_trip'] = (df['trip_days'] >= 10).astype(int)
        features['is_medium_trip'] = ((df['trip_days'] >= 5) & (df['trip_days'] <= 9)).astype(int)
        
        # Binned features
        features['days_bin'] = pd.cut(df['trip_days'], bins=[0, 1, 3, 5, 10, 20], labels=False).fillna(0)
        features['miles_bin'] = pd.cut(df['miles_traveled'], bins=[0, 100, 500, 1000, 2000, 5000], labels=False).fillna(0)
        features['receipts_bin'] = pd.cut(df['total_receipts_amount'], bins=[0, 50, 500, 1000, 2000, 5000], labels=False).fillna(0)
        
        # Rule engine error magnitude
        features['pred_magnitude'] = abs(df['rule_engine_pred'])
        features['pred_log'] = np.log1p(abs(df['rule_engine_pred']))
        
        return features
    
    def predict(self, trip_days, miles_traveled, total_receipts_amount):
        """
        Predict reimbursement using rule engine + ensemble correction.
        
        Args:
            trip_days: Number of days for the trip
            miles_traveled: Miles traveled
            total_receipts_amount: Total receipt amount
            
        Returns:
            Predicted reimbursement amount
        """
        # Get base prediction from rule engine
        base_pred = calculate_reimbursement_v3(trip_days, miles_traveled, total_receipts_amount)
        
        # If no models loaded, return base prediction
        if not self.models:
            return base_pred
        
        # Engineer features
        features = self.engineer_features(trip_days, miles_traveled, total_receipts_amount, base_pred)
        
        # Get residual predictions from each model
        residual_corrections = []
        for model_name, model in self.models.items():
            pred = model.predict(features)[0]
            weighted_pred = pred * self.weights[model_name]
            residual_corrections.append(weighted_pred)
        
        # Sum weighted corrections
        total_correction = sum(residual_corrections)
        
        # Final prediction
        final_pred = base_pred + total_correction
        
        # Ensure non-negative
        final_pred = max(0, final_pred)
        
        # Round to 2 decimal places
        return round(final_pred, 2)


# Convenience function for backward compatibility
def calculate_reimbursement_v4(trip_days, miles_traveled, total_receipts_amount):
    """
    Calculate reimbursement using v4 ensemble-corrected model.
    
    This is a convenience wrapper that creates a singleton instance.
    """
    if not hasattr(calculate_reimbursement_v4, '_model'):
        calculate_reimbursement_v4._model = EnsembleCorrectedModel()
    
    return calculate_reimbursement_v4._model.predict(
        trip_days, miles_traveled, total_receipts_amount
    )


if __name__ == "__main__":
    # Test the model
    print("Testing V4 Ensemble-Corrected Model")
    print("=" * 50)
    
    test_cases = [
        (3, 93, 1.42),    # Expected: 364.51
        (5, 730, 485.73), # Expected: 991.49
        (8, 1200, 1100),  # Special VIP profile
        (1, 600, 1000),   # Single day high miles
        (10, 500, 2000),  # Long trip high receipts
    ]
    
    model = EnsembleCorrectedModel()
    
    for days, miles, receipts in test_cases:
        base_pred = calculate_reimbursement_v3(days, miles, receipts)
        final_pred = model.predict(days, miles, receipts)
        print(f"\nTrip: {days} days, {miles} miles, ${receipts:.2f} receipts")
        print(f"Base prediction: ${base_pred:.2f}")
        print(f"Final prediction: ${final_pred:.2f}")
        print(f"Correction: ${final_pred - base_pred:.2f}") 