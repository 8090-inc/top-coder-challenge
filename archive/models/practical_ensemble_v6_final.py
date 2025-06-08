"""
Practical Ensemble Model v6 Final - Optimized version

This final v6 model:
1. Uses the improved v6 rule engine (MAE $111.72 vs v3's $115.99)
2. Combines it with v3 predictions for robustness
3. Applies targeted ML corrections with strong regularization
4. Maintains the successful patterns from v5
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib
from cluster_models_optimized_v6 import calculate_reimbursement_v6
from cluster_models_optimized import calculate_reimbursement_v3
import os


class FinalEnsembleV6:
    """Final v6 ensemble combining best practices"""
    
    def __init__(self):
        self.models = None
        self.is_trained = False
        self.weights = {'v6': 0.65, 'v3': 0.35}  # Favor improved v6 but keep v3 for stability
        
    def create_features(self, df):
        """Create robust feature set"""
        features = pd.DataFrame()
        
        # Core features
        features['trip_days'] = df['trip_days']
        features['miles'] = df['miles']
        features['receipts'] = df['receipts']
        
        # Rule engine predictions
        features['v6_pred'] = df['v6_pred']
        features['v3_pred'] = df['v3_pred']
        features['rule_diff'] = features['v6_pred'] - features['v3_pred']
        features['rule_avg'] = (features['v6_pred'] + features['v3_pred']) / 2
        
        # Ratios
        features['miles_per_day'] = features['miles'] / features['trip_days']
        features['receipts_per_day'] = features['receipts'] / features['trip_days']
        
        # Log transforms for stability
        features['log_miles'] = np.log1p(features['miles'])
        features['log_receipts'] = np.log1p(features['receipts'])
        
        # Trip type indicators
        features['is_short'] = (features['trip_days'] <= 3).astype(int)
        features['is_medium'] = ((features['trip_days'] > 3) & (features['trip_days'] <= 7)).astype(int)
        features['is_long'] = (features['trip_days'] > 7).astype(int)
        
        # Receipt patterns
        cents = (features['receipts'] * 100).astype(int) % 100
        features['has_49_cents'] = (cents == 49).astype(int)
        features['has_99_cents'] = (cents == 99).astype(int)
        
        return features
    
    def train(self, df):
        """Train the final ensemble"""
        print("Training Final V6 Ensemble Model...")
        
        # Calculate rule engine predictions
        print("Calculating rule engine predictions...")
        v6_preds, v3_preds = [], []
        for _, row in df.iterrows():
            v6_preds.append(calculate_reimbursement_v6(row['trip_days'], row['miles'], row['receipts']))
            v3_preds.append(calculate_reimbursement_v3(row['trip_days'], row['miles'], row['receipts']))
        
        df['v6_pred'] = v6_preds
        df['v3_pred'] = v3_preds
        df['base_ensemble'] = self.weights['v6'] * df['v6_pred'] + self.weights['v3'] * df['v3_pred']
        df['residual'] = df['expected_output'] - df['base_ensemble']
        
        # Show statistics
        print(f"\nBase statistics:")
        print(f"  V6 MAE: ${np.abs(df['expected_output'] - df['v6_pred']).mean():.2f}")
        print(f"  V3 MAE: ${np.abs(df['expected_output'] - df['v3_pred']).mean():.2f}")
        print(f"  Base ensemble MAE: ${np.abs(df['expected_output'] - df['base_ensemble']).mean():.2f}")
        print(f"  Mean residual: ${df['residual'].mean():.2f}")
        print(f"  Std residual: ${df['residual'].std():.2f}")
        
        # Create features
        X = self.create_features(df)
        y = df['residual']
        
        # Define conservative models
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=150,
                max_depth=8,  # Shallow trees
                min_samples_split=30,  # Require many samples
                min_samples_leaf=15,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.03,  # Very conservative
                max_depth=4,
                min_samples_split=30,
                min_samples_leaf=15,
                subsample=0.7,
                random_state=42
            ),
            'ridge': Ridge(alpha=10.0, random_state=42)
        }
        
        # Train models
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X, y)
        
        self.is_trained = True
        
        # Calculate final performance
        final_preds = self.predict_batch(df)
        final_mae = np.abs(df['expected_output'] - final_preds).mean()
        print(f"\nFinal training MAE: ${final_mae:.2f}")
        
        return final_mae
    
    def predict(self, trip_days, miles, receipts):
        """Make a prediction combining v6 and v3 with ML corrections"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get rule engine predictions
        v6_pred = calculate_reimbursement_v6(trip_days, miles, receipts)
        v3_pred = calculate_reimbursement_v3(trip_days, miles, receipts)
        
        # Base ensemble
        base_pred = self.weights['v6'] * v6_pred + self.weights['v3'] * v3_pred
        
        # Create features
        df_single = pd.DataFrame({
            'trip_days': [trip_days],
            'miles': [miles],
            'receipts': [receipts],
            'v6_pred': [v6_pred],
            'v3_pred': [v3_pred]
        })
        
        X = self.create_features(df_single)
        
        # Get ML corrections
        corrections = []
        model_weights = {'rf': 0.4, 'gbm': 0.4, 'ridge': 0.2}
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            corrections.append(pred * model_weights[name])
        
        # Average correction with damping
        correction = sum(corrections) * 0.7  # Conservative 70% damping
        
        # Apply bounds based on prediction magnitude
        max_correction_pct = 0.15  # Max 15% correction
        max_correction = min(60, base_pred * max_correction_pct)
        
        if abs(correction) > max_correction:
            correction = max_correction * np.sign(correction)
        
        # Final prediction
        final_pred = base_pred + correction
        
        # Apply cents patterns (from v5 success)
        # Short trips often end in .29 or .79
        if trip_days <= 2:
            decimal = final_pred % 1
            if 0.20 <= decimal <= 0.35:
                final_pred = int(final_pred) + 0.29
            elif 0.70 <= decimal <= 0.85:
                final_pred = int(final_pred) + 0.79
        
        # Medium trips often end in .49
        elif 3 <= trip_days <= 5 and receipts < 500:
            decimal = final_pred % 1
            if 0.35 <= decimal <= 0.60:
                final_pred = int(final_pred) + 0.49
        
        # High value often ends in .99
        elif final_pred > 1500:
            decimal = final_pred % 1
            if decimal > 0.80:
                final_pred = int(final_pred) + 0.99
        
        return round(max(0, final_pred), 2)
    
    def predict_batch(self, df):
        """Batch predictions"""
        predictions = []
        for _, row in df.iterrows():
            pred = self.predict(row['trip_days'], row['miles'], row['receipts'])
            predictions.append(pred)
        return np.array(predictions)
    
    def save(self, path='v6_final_ensemble.pkl'):
        """Save model"""
        joblib.dump({
            'models': self.models,
            'weights': self.weights
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path='v6_final_ensemble.pkl'):
        """Load model"""
        data = joblib.load(path)
        self.models = data['models']
        self.weights = data['weights']
        self.is_trained = True


def test_final_v6():
    """Test the final v6 model"""
    df = pd.read_csv('../public_cases_expected_outputs.csv')
    
    print("FINAL V6 ENSEMBLE MODEL TEST")
    print("=" * 60)
    
    # Train model
    model = FinalEnsembleV6()
    model.train(df)
    
    # Make predictions
    predictions = model.predict_batch(df)
    df['final_v6_pred'] = predictions
    df['error'] = np.abs(df['final_v6_pred'] - df['expected_output'])
    
    # Metrics
    mae = df['error'].mean()
    bias = (df['final_v6_pred'] - df['expected_output']).mean()
    rmse = np.sqrt(((df['final_v6_pred'] - df['expected_output']) ** 2).mean())
    
    print(f"\nFinal Performance:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Bias: ${bias:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    
    # Comparison
    print(f"\nComparison:")
    print(f"  v5 MAE: $77.41")
    print(f"  v6 MAE: ${mae:.2f}")
    improvement = 77.41 - mae
    print(f"  Improvement: ${improvement:.2f} ({'%.1f' % (improvement / 77.41 * 100)}%)")
    
    # Error distribution
    print(f"\nError Distribution:")
    print(f"  < $50: {(df['error'] < 50).sum()} cases ({(df['error'] < 50).sum() / 10:.1f}%)")
    print(f"  < $100: {(df['error'] < 100).sum()} cases ({(df['error'] < 100).sum() / 10:.1f}%)")
    print(f"  < $200: {(df['error'] < 200).sum()} cases ({(df['error'] < 200).sum() / 10:.1f}%)")
    print(f"  >= $200: {(df['error'] >= 200).sum()} cases")
    
    # Exact matches
    df['exact_match'] = np.abs(df['final_v6_pred'] - df['expected_output']) < 0.01
    print(f"\nExact matches: {df['exact_match'].sum()}/1000")
    
    # Cents pattern analysis
    df['output_cents'] = ((df['expected_output'] * 100) % 100).astype(int)
    df['pred_cents'] = ((df['final_v6_pred'] * 100) % 100).astype(int)
    df['cents_match'] = df['output_cents'] == df['pred_cents']
    print(f"Cents pattern matches: {df['cents_match'].sum()}/1000")
    
    # Save model and predictions
    model.save()
    
    # Create predictions directory if needed
    os.makedirs('../predictions', exist_ok=True)
    df[['expected_output', 'final_v6_pred', 'error']].to_csv('../predictions/v6_final_predictions.csv', index=False)
    print(f"\nPredictions saved to predictions/v6_final_predictions.csv")
    
    return df, model


# Convenience function for easy use
def calculate_reimbursement_v6_final(trip_days, miles, receipts):
    """Calculate using final v6 ensemble"""
    if not hasattr(calculate_reimbursement_v6_final, '_model'):
        model = FinalEnsembleV6()
        try:
            model.load('v6_final_ensemble.pkl')
        except:
            raise ValueError("Model not found. Please run test_final_v6() first.")
        calculate_reimbursement_v6_final._model = model
    
    return calculate_reimbursement_v6_final._model.predict(trip_days, miles, receipts)


if __name__ == "__main__":
    df, model = test_final_v6() 