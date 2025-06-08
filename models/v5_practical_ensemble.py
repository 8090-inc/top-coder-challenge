"""
V5 Practical Ensemble Model - Production Ready

A pragmatic approach that:
1. Uses all 1000 samples for training (no holdout)
2. Simple but robust ensemble with strong regularization
3. Conservative corrections to avoid overfitting
4. Ready for cents hash integration when discovered
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

# Import v3 rule engine
try:
    from cluster_models_optimized import calculate_reimbursement_v3
except ImportError:
    from models.cluster_models_optimized import calculate_reimbursement_v3


class PracticalEnsembleModel:
    """Production-ready ensemble with conservative corrections"""
    
    def __init__(self, max_correction=100):
        """
        Initialize with conservative settings.
        
        Args:
            max_correction: Maximum allowed correction (default $100)
        """
        self.max_correction = max_correction
        self.models = {}
        self.is_trained = False
        self.training_stats = {}
        
    def engineer_features(self, df):
        """Simplified feature engineering for robustness"""
        features = pd.DataFrame()
        
        # Core features
        features['trip_days'] = df['trip_days']
        features['miles'] = df['miles_traveled']
        features['receipts'] = df['total_receipts_amount']
        features['rule_pred'] = df['rule_engine_pred']
        
        # Simple derived features
        features['miles_per_day'] = features['miles'] / features['trip_days']
        features['receipts_per_day'] = features['receipts'] / features['trip_days']
        
        # Log transforms for stability
        features['log_miles'] = np.log1p(features['miles'])
        features['log_receipts'] = np.log1p(features['receipts'])
        
        # Binary flags
        features['is_single_day'] = (features['trip_days'] == 1).astype(int)
        features['is_long_trip'] = (features['trip_days'] >= 10).astype(int)
        features['has_low_receipts'] = (features['receipts'] < 50).astype(int)
        features['has_49_ending'] = ((features['receipts'] * 100).astype(int) % 100 == 49).astype(int)
        features['has_99_ending'] = ((features['receipts'] * 100).astype(int) % 100 == 99).astype(int)
        
        return features
    
    def train(self, train_df):
        """Train ensemble on full dataset"""
        print("Training Practical Ensemble Model v5...")
        
        # Generate rule engine predictions
        print("Generating rule engine predictions...")
        rule_preds = []
        for _, row in train_df.iterrows():
            pred = calculate_reimbursement_v3(
                row['trip_days'],
                row['miles_traveled'], 
                row['total_receipts_amount']
            )
            rule_preds.append(pred)
        
        train_df['rule_engine_pred'] = rule_preds
        train_df['residual'] = train_df['expected_reimbursement'] - train_df['rule_engine_pred']
        
        # Prepare features
        X = self.engineer_features(train_df)
        y = train_df['residual']
        
        # Store training statistics
        self.training_stats = {
            'n_samples': len(train_df),
            'residual_mean': float(y.mean()),
            'residual_std': float(y.std()),
            'residual_95_pct': float(np.percentile(np.abs(y), 95))
        }
        
        print(f"\nTraining statistics:")
        print(f"  Samples: {self.training_stats['n_samples']}")
        print(f"  Mean residual: ${self.training_stats['residual_mean']:.2f}")
        print(f"  Residual std: ${self.training_stats['residual_std']:.2f}")
        print(f"  95th percentile: ${self.training_stats['residual_95_pct']:.2f}")
        
        # Define conservative models
        self.models = {
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=8,  # Limit depth
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=10,   # Larger leaves
                max_features='sqrt',   # Use subset of features
                random_state=42,
                n_jobs=-1
            ),
            'GBM': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.02,   # Very low learning rate
                max_depth=5,          # Shallow trees
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.7,
                max_features='sqrt',
                random_state=42
            )
        }
        
        # Train models
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X, y)
            
            # In-sample performance (for monitoring)
            train_pred = model.predict(X)
            train_mae = np.mean(np.abs(y - train_pred))
            print(f"    In-sample MAE: ${train_mae:.2f}")
        
        self.is_trained = True
        print("\nTraining complete!")
        
        # Return final MAE for monitoring
        final_preds = self.predict_batch(train_df)
        final_mae = np.mean(np.abs(train_df['expected_reimbursement'] - final_preds))
        print(f"\nFinal training MAE: ${final_mae:.2f}")
        
        return final_mae
    
    def predict(self, trip_days, miles_traveled, total_receipts_amount):
        """Predict single case with conservative correction"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get rule engine prediction
        rule_pred = calculate_reimbursement_v3(trip_days, miles_traveled, total_receipts_amount)
        
        # Create features
        df_single = pd.DataFrame({
            'trip_days': [trip_days],
            'miles_traveled': [miles_traveled],
            'total_receipts_amount': [total_receipts_amount],
            'rule_engine_pred': [rule_pred]
        })
        
        features = self.engineer_features(df_single)
        
        # Get predictions from each model
        corrections = []
        weights = {'ExtraTrees': 0.5, 'GBM': 0.5}
        
        for name, model in self.models.items():
            pred = model.predict(features)[0]
            corrections.append(pred * weights[name])
        
        # Average correction
        correction = sum(corrections)
        
        # Apply conservative bounds
        # Scale down large corrections
        if abs(correction) > self.max_correction:
            scale_factor = self.max_correction / abs(correction)
            correction *= scale_factor
        
        # Further reduce if correction is large relative to training distribution
        if 'residual_95_pct' in self.training_stats:
            pct_95 = self.training_stats['residual_95_pct']
            if abs(correction) > pct_95:
                # Soft scaling for extreme corrections
                excess = abs(correction) - pct_95
                scale = 1 - (excess / (excess + pct_95)) * 0.5
                correction *= scale
        
        # Final prediction
        final_pred = rule_pred + correction
        
        # Ensure non-negative
        final_pred = max(0, final_pred)
        
        return round(final_pred, 2)
    
    def predict_batch(self, df):
        """Predict multiple cases efficiently"""
        predictions = []
        for _, row in df.iterrows():
            pred = self.predict(
                row['trip_days'],
                row['miles_traveled'],
                row['total_receipts_amount']
            )
            predictions.append(pred)
        return np.array(predictions)
    
    def save(self, path='v5_practical_ensemble.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        model_data = {
            'models': self.models,
            'training_stats': self.training_stats,
            'max_correction': self.max_correction
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load(self, path='v5_practical_ensemble.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        
        self.models = model_data['models']
        self.training_stats = model_data['training_stats']
        self.max_correction = model_data['max_correction']
        self.is_trained = True
        
        print(f"Model loaded from {path}")


# Convenience function
def calculate_reimbursement_v5(trip_days, miles_traveled, total_receipts_amount):
    """Calculate using v5 practical ensemble"""
    if not hasattr(calculate_reimbursement_v5, '_model'):
        # Load or create model
        model = PracticalEnsembleModel()
        model_path = Path(__file__).parent / 'v5_practical_ensemble.pkl'
        
        if model_path.exists():
            model.load(str(model_path))
        else:
            # Need to train first
            raise ValueError("Model not found. Please train first using train_v5_model.py")
        
        calculate_reimbursement_v5._model = model
    
    return calculate_reimbursement_v5._model.predict(
        trip_days, miles_traveled, total_receipts_amount
    )


if __name__ == "__main__":
    # Quick test
    print("Testing V5 Practical Ensemble Model")
    print("=" * 50)
    
    # Create a dummy model for testing structure
    model = PracticalEnsembleModel()
    print("Model initialized successfully!")
    print(f"Max correction: ${model.max_correction}")
    print("\nTo train: use train_v5_model.py") 