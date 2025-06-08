"""
V5.1 Practical Ensemble Model - With Fixed Receipt Penalties

This is v5 with the corrected receipt penalty factors:
- .99 endings: 0.745 (was 0.51)
- .49 endings: 0.455 (was 0.341)
"""

from models.v5_practical_ensemble import PracticalEnsembleModel
from models.cluster_models_optimized_fixed import calculate_reimbursement_v3_fixed
import numpy as np
import pandas as pd

class PracticalEnsembleModelV51(PracticalEnsembleModel):
    """V5.1 with fixed receipt penalties"""
    
    def train(self, train_df):
        """Train ensemble on full dataset using fixed rule engine"""
        print("Training Practical Ensemble Model v5.1 (Fixed Penalties)...")
        
        # Generate rule engine predictions with FIXED penalties
        print("Generating rule engine predictions with fixed penalties...")
        rule_preds = []
        for _, row in train_df.iterrows():
            pred = calculate_reimbursement_v3_fixed(
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
        
        # Train models (using parent class models)
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
        """Predict single case with fixed penalties"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get rule engine prediction with FIXED penalties
        rule_pred = calculate_reimbursement_v3_fixed(trip_days, miles_traveled, total_receipts_amount)
        
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
        if abs(correction) > self.max_correction:
            scale_factor = self.max_correction / abs(correction)
            correction *= scale_factor
        
        # Further reduce if correction is large relative to training distribution
        if 'residual_95_pct' in self.training_stats:
            pct_95 = self.training_stats['residual_95_pct']
            if abs(correction) > pct_95:
                excess = abs(correction) - pct_95
                scale = 1 - (excess / (excess + pct_95)) * 0.5
                correction *= scale
        
        # Final prediction
        final_pred = rule_pred + correction
        
        # Ensure non-negative
        final_pred = max(0, final_pred)
        
        return round(final_pred, 2) 