"""
V6 Advanced Ensemble Model
Incorporates:
1. Per-cluster correction models (ET + GBM)
2. Dynamic clip bounds based on cluster percentiles
3. Receipt-cents and miles % 64 features
4. Outlier guard for extreme corrections
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
from typing import Dict, Tuple, Any

from cluster_models_optimized import calculate_reimbursement_v3
from cluster_router import assign_cluster_v2


class ClusterModelsV3Wrapper:
    """Wrapper to make the functional API compatible with class-based interface"""
    def fit(self, X, y):
        # No fitting needed for rule-based model
        pass
    
    def predict(self, X):
        predictions = []
        for idx in range(len(X)):
            pred = calculate_reimbursement_v3(
                X.iloc[idx]['trip_days'],
                X.iloc[idx]['miles_traveled'],
                X.iloc[idx]['total_receipts_amount']
            )
            predictions.append(pred)
        return np.array(predictions)


class V6AdvancedEnsemble:
    """
    V6 model with advanced per-cluster residual correction
    """
    
    def __init__(self):
        self.base_model = ClusterModelsV3Wrapper()
        self.router = None  # We'll use assign_cluster_v2 directly
        
        # Per-cluster correction models
        self.et_models = {}  # Extra Trees per cluster
        self.gbm_models = {}  # Gradient Boosting per cluster
        
        # Dynamic clip bounds per cluster
        self.clip_bounds = {}
        
        # Model parameters
        self.et_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.gbm_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Feature engineering settings
        self.use_advanced_features = True
    
    def _get_cluster_assignments(self, X: pd.DataFrame) -> np.ndarray:
        """Get numeric cluster assignments for all samples"""
        clusters = []
        for idx in range(len(X)):
            cluster_str = assign_cluster_v2(
                X.iloc[idx]['trip_days'],
                X.iloc[idx]['miles_traveled'], 
                X.iloc[idx]['total_receipts_amount']
            )
            # Convert string clusters to numeric
            cluster_map = {
                '0': 0, '0_low_mile_high_receipt': 0,
                '1a': 1, '1b': 1,
                '2': 2, '3': 3, '4': 4, '5': 5, '6': 6
            }
            clusters.append(cluster_map.get(cluster_str, 0))
        return np.array(clusters)
        
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features including receipt cents and miles modulo
        """
        X_feat = X.copy()
        
        # Basic features
        X_feat['days_squared'] = X_feat['trip_days'] ** 2
        X_feat['miles_per_day'] = X_feat['miles_traveled'] / X_feat['trip_days']
        X_feat['receipts_per_day'] = X_feat['total_receipts_amount'] / X_feat['trip_days']
        
        # Advanced features for hash discovery
        if self.use_advanced_features:
            # Receipt cents (last 2 digits of receipts)
            X_feat['centsR'] = (X_feat['total_receipts_amount'] * 100).astype(int) % 100
            
            # Miles modulo 64 (for potential hash patterns)
            X_feat['miles_mod_64'] = X_feat['miles_traveled'].astype(int) % 64
            
            # Additional modulo features that might be useful
            X_feat['miles_mod_100'] = X_feat['miles_traveled'].astype(int) % 100
            X_feat['days_mod_7'] = X_feat['trip_days'] % 7
            
            # Interaction features
            X_feat['cents_miles_interact'] = X_feat['centsR'] * X_feat['miles_mod_64']
            
        # Handle any infinities or NaNs
        X_feat = X_feat.replace([np.inf, -np.inf], 0).fillna(0)
        
        return X_feat
    
    def _get_cluster_clip_bounds(self, residuals: np.ndarray, cluster: int) -> Tuple[float, float]:
        """
        Calculate dynamic clip bounds for a cluster based on residual distribution
        """
        # Different percentiles for different clusters
        percentile_map = {
            0: 95,    # Standard multi-day: moderate clipping
            1: 90,    # Single day high miles: tighter clipping  
            2: 97,    # Long trip high receipts: looser clipping
            3: 95,    # Short trip high expenses
            4: 85,    # Outlier cluster: tight clipping
            5: 99,    # Medium trip high miles: very loose (VIP)
            6: 90,    # 1-day trips
            7: 85,    # High miles low receipts: tight
            8: 92     # 1-day low miles
        }
        
        percentile = percentile_map.get(cluster, 95)
        
        # Calculate bounds
        lower_bound = np.percentile(residuals, 100 - percentile)
        upper_bound = np.percentile(residuals, percentile)
        
        # Ensure minimum bounds
        min_bound = 50  # At least $50 correction allowed
        lower_bound = min(lower_bound, -min_bound)
        upper_bound = max(upper_bound, min_bound)
        
        return lower_bound, upper_bound
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the V6 model with per-cluster correction models
        """
        print("Training V6 Advanced Ensemble Model...")
        
        # First, fit the base model
        print("Fitting base cluster models...")
        self.base_model.fit(X, y)
        
        # Get base predictions and residuals
        base_preds = self.base_model.predict(X)
        residuals = y - base_preds
        
        # Get cluster assignments
        clusters = self._get_cluster_assignments(X)
        
        # Engineer features
        X_feat = self._engineer_features(X)
        
        # Train per-cluster correction models
        print("\nTraining per-cluster correction models...")
        
        for cluster in range(9):  # 9 clusters total
            cluster_mask = clusters == cluster
            n_samples = cluster_mask.sum()
            
            if n_samples < 10:  # Skip if too few samples
                print(f"Cluster {cluster}: Too few samples ({n_samples}), skipping")
                continue
                
            print(f"\nCluster {cluster}: {n_samples} samples")
            
            # Get cluster data
            X_cluster = X_feat[cluster_mask]
            residuals_cluster = residuals[cluster_mask]
            
            # Calculate and store clip bounds
            self.clip_bounds[cluster] = self._get_cluster_clip_bounds(residuals_cluster, cluster)
            print(f"  Clip bounds: [{self.clip_bounds[cluster][0]:.2f}, {self.clip_bounds[cluster][1]:.2f}]")
            
            # Split for validation
            if n_samples > 50:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_cluster, residuals_cluster, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train = X_cluster, residuals_cluster
                X_val, y_val = None, None
            
            # Train Extra Trees
            print(f"  Training Extra Trees...")
            self.et_models[cluster] = ExtraTreesRegressor(**self.et_params)
            self.et_models[cluster].fit(X_train, y_train)
            
            # Train Gradient Boosting
            print(f"  Training Gradient Boosting...")
            self.gbm_models[cluster] = GradientBoostingRegressor(**self.gbm_params)
            self.gbm_models[cluster].fit(X_train, y_train)
            
            # Validate if we have validation data
            if X_val is not None:
                et_pred = self.et_models[cluster].predict(X_val)
                gbm_pred = self.gbm_models[cluster].predict(X_val)
                ensemble_pred = 0.6 * et_pred + 0.4 * gbm_pred
                val_mae = np.mean(np.abs(y_val - ensemble_pred))
                print(f"  Validation MAE: ${val_mae:.2f}")
        
        print("\nV6 model training complete!")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with per-cluster correction and outlier guard
        """
        # Get base predictions
        base_preds = self.base_model.predict(X)
        
        # Get cluster assignments
        clusters = self._get_cluster_assignments(X)
        
        # Engineer features
        X_feat = self._engineer_features(X)
        
        # Apply per-cluster corrections
        corrections = np.zeros(len(X))
        
        for i in range(len(X)):
            cluster = clusters[i]
            
            # Check if we have models for this cluster
            if cluster not in self.et_models:
                continue
            
            # Get correction from ensemble
            X_single = X_feat.iloc[[i]]
            et_correction = self.et_models[cluster].predict(X_single)[0]
            gbm_correction = self.gbm_models[cluster].predict(X_single)[0]
            
            # Ensemble correction (60% ET, 40% GBM)
            correction = 0.6 * et_correction + 0.4 * gbm_correction
            
            # Apply dynamic clip bounds
            if cluster in self.clip_bounds:
                lower, upper = self.clip_bounds[cluster]
                correction = np.clip(correction, lower, upper)
            
            corrections[i] = correction
        
        # Apply corrections to base predictions
        final_preds = base_preds + corrections
        
        # Outlier guard: prevent extreme final predictions
        # No prediction should exceed $400 in correction from base
        max_correction = 400
        correction_magnitude = np.abs(final_preds - base_preds)
        outlier_mask = correction_magnitude > max_correction
        
        if outlier_mask.any():
            print(f"Warning: Capping {outlier_mask.sum()} extreme corrections")
            # Cap the corrections
            for i in np.where(outlier_mask)[0]:
                if final_preds[i] > base_preds[i]:
                    final_preds[i] = base_preds[i] + max_correction
                else:
                    final_preds[i] = base_preds[i] - max_correction
        
        # Ensure non-negative predictions
        final_preds = np.maximum(final_preds, 0)
        
        return final_preds
    
    def save(self, filepath: str):
        """Save the model to disk"""
        model_data = {
            'clip_bounds': self.clip_bounds,
            'et_models': self.et_models,
            'gbm_models': self.gbm_models
        }
        joblib.dump(model_data, filepath)
        print(f"V6 model saved to {filepath}")
        
    def load(self, filepath: str):
        """Load the model from disk"""
        model_data = joblib.load(filepath)
        self.clip_bounds = model_data['clip_bounds']
        self.et_models = model_data['et_models']
        self.gbm_models = model_data['gbm_models']
        print(f"V6 model loaded from {filepath}")
        

def evaluate_model(model, X: pd.DataFrame, y: np.ndarray, name: str = "Model"):
    """Evaluate model performance"""
    predictions = model.predict(X)
    residuals = y - predictions
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    max_error = np.max(np.abs(residuals))
    
    print(f"\n{name} Performance:")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"Max Error: ${max_error:.2f}")
    print(f"Errors > $500: {np.sum(np.abs(residuals) > 500)}")
    
    return mae, predictions 