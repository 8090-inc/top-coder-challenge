"""
Practical Ensemble Model v6 - Improved version with proper ML integration

This model improves on v5 by:
1. Using the improved v6 rule engine (with Cluster 0 optimization)
2. Training a residual correction model specifically for v6 predictions
3. Maintaining conservative bounds to avoid overfitting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
import joblib
from cluster_models_optimized_v6 import calculate_reimbursement_v6
from cluster_models_optimized import calculate_reimbursement_v3


class EnsembleV6:
    """Ensemble model combining v6 rule engine with ML corrections"""
    
    def __init__(self):
        self.models = None
        self.is_trained = False
        self.training_mae = None
        
    def create_features(self, df, include_v6_pred=True):
        """Create features for ML models"""
        features = pd.DataFrame()
        
        # Basic features
        features['trip_days'] = df['trip_days']
        features['miles'] = df['miles'] 
        features['receipts'] = df['receipts']
        
        # Ratios
        features['miles_per_day'] = features['miles'] / features['trip_days']
        features['receipts_per_day'] = features['receipts'] / features['trip_days']
        features['receipts_per_mile'] = features['receipts'] / (features['miles'] + 1)
        
        # Log transforms
        features['log_miles'] = np.log1p(features['miles'])
        features['log_receipts'] = np.log1p(features['receipts'])
        features['log_trip_days'] = np.log1p(features['trip_days'])
        
        # Polynomial features
        features['trip_days_squared'] = features['trip_days'] ** 2
        features['miles_receipts_interaction'] = features['miles'] * features['receipts'] / 10000
        
        # Categorical indicators
        features['is_short_trip'] = (features['trip_days'] <= 3).astype(int)
        features['is_long_trip'] = (features['trip_days'] >= 10).astype(int)
        features['is_high_miles'] = (features['miles'] > 1000).astype(int)
        features['is_high_receipts'] = (features['receipts'] > 1000).astype(int)
        
        # Receipt ending patterns
        features['receipts_49_ending'] = ((features['receipts'] * 100) % 100 == 49).astype(int)
        features['receipts_99_ending'] = ((features['receipts'] * 100) % 100 == 99).astype(int)
        
        if include_v6_pred:
            features['v6_pred'] = df['v6_pred']
            features['v6_pred_per_day'] = features['v6_pred'] / features['trip_days']
            
        return features
    
    def train(self, df):
        """Train the ensemble on the data"""
        print("Training V6 Ensemble Model...")
        
        # Calculate v6 predictions
        print("Calculating v6 rule engine predictions...")
        v6_preds = []
        for _, row in df.iterrows():
            pred = calculate_reimbursement_v6(row['trip_days'], row['miles'], row['receipts'])
            v6_preds.append(pred)
        
        df['v6_pred'] = v6_preds
        df['residual'] = df['expected_output'] - df['v6_pred']
        
        # Create features
        X = self.create_features(df)
        y = df['residual']
        
        print(f"\nTraining statistics:")
        print(f"  Mean residual: ${y.mean():.2f}")
        print(f"  Std residual: ${y.std():.2f}")
        print(f"  V6 MAE: ${np.abs(df['expected_output'] - df['v6_pred']).mean():.2f}")
        
        # Define models with strong regularization
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42
            ),
            'et': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42
            ),
            'huber': HuberRegressor(
                alpha=0.1,
                max_iter=200,
                epsilon=1.5
            )
        }
        
        # Train models
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X, y)
        
        # Mark as trained before calculating performance
        self.is_trained = True
        
        # Calculate training performance
        train_preds = self.predict_batch(df)
        self.training_mae = np.abs(df['expected_output'] - train_preds).mean()
        
        print(f"\nFinal training MAE: ${self.training_mae:.2f}")
        
        return self.training_mae
    
    def predict(self, trip_days, miles, receipts):
        """Make a single prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get v6 rule engine prediction
        v6_pred = calculate_reimbursement_v6(trip_days, miles, receipts)
        
        # Create features
        df_single = pd.DataFrame({
            'trip_days': [trip_days],
            'miles': [miles],
            'receipts': [receipts],
            'v6_pred': [v6_pred]
        })
        
        X = self.create_features(df_single)
        
        # Get predictions from each model
        corrections = []
        weights = {'rf': 0.4, 'et': 0.4, 'huber': 0.2}
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            corrections.append(pred * weights[name])
        
        # Average correction
        correction = sum(corrections)
        
        # Apply conservative bounds
        max_correction = 80  # Maximum $80 correction
        if abs(correction) > max_correction:
            correction = max_correction * np.sign(correction)
        
        # Further damping for extreme corrections
        if abs(correction) > 50:
            damping = 0.7
            correction *= damping
        
        # Final prediction
        final_pred = v6_pred + correction
        
        # Apply cents patterns for common endings
        # .29 pattern for short trips
        if trip_days <= 2 and 0.20 <= (final_pred % 1) <= 0.35:
            final_pred = int(final_pred) + 0.29
        # .49 pattern for medium trips
        elif 3 <= trip_days <= 5 and 0.40 <= (final_pred % 1) <= 0.55:
            final_pred = int(final_pred) + 0.49
        # .99 pattern for high value
        elif final_pred > 1200 and (final_pred % 1) > 0.85:
            final_pred = int(final_pred) + 0.99
        
        return round(max(0, final_pred), 2)
    
    def predict_batch(self, df):
        """Predict multiple cases"""
        predictions = []
        for _, row in df.iterrows():
            pred = self.predict(row['trip_days'], row['miles'], row['receipts'])
            predictions.append(pred)
        return np.array(predictions)
    
    def save(self, path='v6_ensemble.pkl'):
        """Save the model"""
        joblib.dump({
            'models': self.models,
            'training_mae': self.training_mae
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path='v6_ensemble.pkl'):
        """Load the model"""
        data = joblib.load(path)
        self.models = data['models']
        self.training_mae = data['training_mae']
        self.is_trained = True
        print(f"Model loaded from {path}")


def test_v6_improved():
    """Test the improved v6 ensemble"""
    # Load data
    df = pd.read_csv('../public_cases_expected_outputs.csv')
    
    print("V6 IMPROVED ENSEMBLE MODEL TEST")
    print("=" * 60)
    
    # Create and train model
    model = EnsembleV6()
    model.train(df)
    
    # Make predictions
    predictions = model.predict_batch(df)
    df['v6_improved_pred'] = predictions
    df['error'] = np.abs(df['v6_improved_pred'] - df['expected_output'])
    
    # Calculate metrics
    mae = df['error'].mean()
    bias = (df['v6_improved_pred'] - df['expected_output']).mean()
    
    print(f"\nFinal Performance:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Bias: ${bias:.2f}")
    print(f"  vs v5 MAE ($77.41): {'%.1f' % ((77.41 - mae) / 77.41 * 100)}% improvement")
    
    # Error distribution
    print(f"\nError Distribution:")
    print(f"  < $50: {(df['error'] < 50).sum()} cases")
    print(f"  < $100: {(df['error'] < 100).sum()} cases")
    print(f"  < $200: {(df['error'] < 200).sum()} cases")
    print(f"  >= $200: {(df['error'] >= 200).sum()} cases")
    
    # Save model
    model.save()
    
    return df, model


# Convenience function
def calculate_reimbursement_v6_improved(trip_days, miles, receipts):
    """Calculate using v6 improved ensemble"""
    if not hasattr(calculate_reimbursement_v6_improved, '_model'):
        model = EnsembleV6()
        try:
            model.load('v6_ensemble.pkl')
        except:
            # Train if not available
            df = pd.read_csv('../public_cases_expected_outputs.csv')
            model.train(df)
            model.save()
        calculate_reimbursement_v6_improved._model = model
    
    return calculate_reimbursement_v6_improved._model.predict(trip_days, miles, receipts)


if __name__ == "__main__":
    df, model = test_v6_improved() 