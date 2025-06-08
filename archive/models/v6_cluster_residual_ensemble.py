"""
V6 Cluster Residual Ensemble Model
Implements per-cluster residual stacks with adaptive clipping bounds
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

# Import v3 rule engine
try:
    from cluster_models_optimized import calculate_reimbursement_v3, assign_cluster
except ImportError:
    from models.cluster_models_optimized import calculate_reimbursement_v3, assign_cluster


class ClusterResidualEnsemble:
    """Per-cluster residual stacks with adaptive clipping"""
    
    def __init__(self):
        self.cluster_models = {}  # One ensemble per cluster
        self.cluster_bounds = {}  # Adaptive clip bounds per cluster
        self.is_trained = False
        self.training_stats = {}
        
    def engineer_features(self, df):
        """Enhanced feature engineering with rule-delta"""
        features = pd.DataFrame()
        
        # Core features
        features['trip_days'] = df['trip_days']
        features['miles'] = df['miles_traveled']
        features['receipts'] = df['total_receipts_amount']
        features['rule_pred'] = df['rule_engine_pred']
        
        # Rule-delta feature (bullet #3)
        if 'expected_reimbursement' in df.columns:
            features['rule_delta'] = np.abs(df['rule_engine_pred'] - df['expected_reimbursement'])
            features['log_rule_delta'] = np.log1p(features['rule_delta'])
        else:
            # During prediction, we don't have expected values, so use placeholder
            features['rule_delta'] = 0
            features['log_rule_delta'] = 0
        
        # Simple derived features
        features['miles_per_day'] = features['miles'] / np.maximum(features['trip_days'], 1)
        features['receipts_per_day'] = features['receipts'] / np.maximum(features['trip_days'], 1)
        
        # Log transforms (bullet #4)
        features['log_miles'] = np.log1p(features['miles'])
        features['log_receipts'] = np.log1p(features['receipts'])
        features['log_days'] = np.log1p(features['trip_days'])
        
        # Interaction terms (bullet #4)
        features['days_x_log_receipts'] = features['trip_days'] * features['log_receipts']
        features['receipts_squared'] = features['receipts'] ** 2
        
        # Binary flags
        features['is_single_day'] = (features['trip_days'] == 1).astype(int)
        features['is_long_trip'] = (features['trip_days'] >= 10).astype(int)
        features['has_low_receipts'] = (features['receipts'] < 50).astype(int)
        features['has_49_ending'] = ((features['receipts'] * 100).astype(int) % 100 == 49).astype(int)
        features['has_99_ending'] = ((features['receipts'] * 100).astype(int) % 100 == 99).astype(int)
        
        # Cluster assignment
        features['cluster'] = df['cluster']
        
        return features
    
    def train_cluster_model(self, cluster_id, cluster_df):
        """Train ensemble for a specific cluster"""
        X = self.engineer_features(cluster_df)
        y = cluster_df['residual']
        
        # Define conservative models for this cluster
        models = {
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=150,
                max_depth=6,  # Limited depth
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=42 + cluster_id,
                n_jobs=-1
            ),
            'GBM': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=5,
                min_samples_split=15,
                min_samples_leaf=8,
                subsample=0.7,
                max_features='sqrt',
                random_state=42 + cluster_id
            )
        }
        
        # Train models
        trained_models = {}
        predictions = []
        
        for name, model in models.items():
            model.fit(X, y)
            pred = model.predict(X)
            predictions.append(pred)
            trained_models[name] = model
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Calculate adaptive bound (bullet #2)
        # Use 95th percentile of absolute residuals
        residual_errors = np.abs(y - ensemble_pred)
        adaptive_bound = np.percentile(residual_errors, 95)
        
        # Store cluster-specific info
        cluster_stats = {
            'n_samples': len(cluster_df),
            'residual_mean': float(y.mean()),
            'residual_std': float(y.std()),
            'adaptive_bound': float(adaptive_bound),
            'in_sample_mae': float(np.mean(residual_errors))
        }
        
        print(f"  Cluster {cluster_id}: n={cluster_stats['n_samples']}, "
              f"bound=${adaptive_bound:.2f}, MAE=${cluster_stats['in_sample_mae']:.2f}")
        
        return trained_models, adaptive_bound, cluster_stats
    
    def train(self, train_df):
        """Train per-cluster ensembles"""
        print("Training V6 Cluster Residual Ensemble...")
        
        # Generate rule engine predictions and cluster assignments
        print("Generating rule engine predictions...")
        rule_preds = []
        clusters = []
        
        for _, row in train_df.iterrows():
            pred = calculate_reimbursement_v3(
                row['trip_days'],
                row['miles_traveled'], 
                row['total_receipts_amount']
            )
            cluster = assign_cluster(
                row['trip_days'],
                row['miles_traveled'],
                row['total_receipts_amount']
            )
            rule_preds.append(pred)
            clusters.append(cluster)
        
        train_df['rule_engine_pred'] = rule_preds
        train_df['cluster'] = clusters
        train_df['residual'] = train_df['expected_reimbursement'] - train_df['rule_engine_pred']
        
        # Train separate model for each cluster
        print("\nTraining per-cluster models...")
        unique_clusters = sorted(train_df['cluster'].unique())
        
        for cluster_id in unique_clusters:
            cluster_df = train_df[train_df['cluster'] == cluster_id].copy()
            
            if len(cluster_df) < 10:
                # Too few samples - use global stats
                print(f"  Cluster {cluster_id}: Only {len(cluster_df)} samples - skipping")
                continue
            
            models, bound, stats = self.train_cluster_model(cluster_id, cluster_df)
            self.cluster_models[cluster_id] = models
            self.cluster_bounds[cluster_id] = bound
            self.training_stats[cluster_id] = stats
        
        self.is_trained = True
        
        # Calculate final training performance
        final_preds = self.predict_batch(train_df)
        final_mae = np.mean(np.abs(train_df['expected_reimbursement'] - final_preds))
        print(f"\nFinal training MAE: ${final_mae:.2f}")
        
        return final_mae
    
    def predict(self, trip_days, miles_traveled, total_receipts_amount):
        """Predict single case with cluster-specific model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get rule engine prediction and cluster
        rule_pred = calculate_reimbursement_v3(trip_days, miles_traveled, total_receipts_amount)
        cluster = assign_cluster(trip_days, miles_traveled, total_receipts_amount)
        
        # Check if we have a model for this cluster
        if cluster not in self.cluster_models:
            # Fallback to rule engine if no cluster model
            return round(rule_pred, 2)
        
        # Create features
        df_single = pd.DataFrame({
            'trip_days': [trip_days],
            'miles_traveled': [miles_traveled],
            'total_receipts_amount': [total_receipts_amount],
            'rule_engine_pred': [rule_pred],
            'cluster': [cluster]
        })
        
        features = self.engineer_features(df_single)
        
        # Get predictions from cluster-specific models
        corrections = []
        for name, model in self.cluster_models[cluster].items():
            pred = model.predict(features)[0]
            corrections.append(pred)
        
        # Average correction
        correction = np.mean(corrections)
        
        # Apply adaptive clipping bound for this cluster
        bound = self.cluster_bounds[cluster]
        correction = np.clip(correction, -bound, bound)
        
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
    
    def cross_validate(self, df, n_splits=5):
        """5-fold cross validation"""
        print(f"\nRunning {n_splits}-fold cross validation...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Split data
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            
            # Create new instance and train
            fold_model = ClusterResidualEnsemble()
            fold_model.train(train_fold)
            
            # Validate
            val_preds = fold_model.predict_batch(val_fold)
            val_mae = np.mean(np.abs(val_fold['expected_reimbursement'] - val_preds))
            fold_maes.append(val_mae)
            
            print(f"Fold {fold + 1} validation MAE: ${val_mae:.2f}")
        
        avg_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        print(f"\nCross-validation results:")
        print(f"Average MAE: ${avg_mae:.2f} (±${std_mae:.2f})")
        
        return avg_mae, std_mae, fold_maes
    
    def save(self, path='v6_cluster_residual_ensemble.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        model_data = {
            'cluster_models': self.cluster_models,
            'cluster_bounds': self.cluster_bounds,
            'training_stats': self.training_stats
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load(self, path='v6_cluster_residual_ensemble.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        
        self.cluster_models = model_data['cluster_models']
        self.cluster_bounds = model_data['cluster_bounds']
        self.training_stats = model_data['training_stats']
        self.is_trained = True
        
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    print("V6 Cluster Residual Ensemble Model")
    print("=" * 50)
    print("Features:")
    print("- Per-cluster residual stacks")
    print("- Adaptive clipping bounds (95th percentile)")
    print("- Rule-delta feature")
    print("- Enhanced feature engineering")
    print("\nTo train: use train_v6_model.py") 