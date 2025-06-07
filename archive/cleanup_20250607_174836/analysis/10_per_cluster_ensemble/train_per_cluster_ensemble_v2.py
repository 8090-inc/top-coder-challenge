#!/usr/bin/env python3
"""
Per-Cluster Ensemble Training v2 - Improved handling of edge cases

Improvements:
1. Merge small clusters into parent clusters
2. More conservative bounds for small clusters
3. Special handling for outlier cluster (0_low_mile_high_receipt)
4. Adaptive complexity based on cluster size
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('../../data/train.csv')

# Sprint B: Create proper 15% blind test set
print("\n=== Creating 15% blind test set ===")
train_main, train_blind = train_test_split(train_df, test_size=0.15, random_state=42)
print(f"Training set: {len(train_main)} rows")
print(f"Blind test set: {len(train_blind)} rows")

# Load v3 rule engine
print("\nLoading v3 rule engine...")
import sys
sys.path.append('../..')
from models.cluster_models_optimized import calculate_reimbursement_v3 as predict_reimbursement

# Feature engineering (same as before)
def engineer_features(df):
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
    
    # Polynomial features (reduced for stability)
    features['days_squared'] = df['trip_days'] ** 2
    features['receipts_squared'] = np.log1p(df['total_receipts_amount'] ** 2)  # Log to reduce scale
    
    # Key interactions only
    features['days_x_receipts'] = df['trip_days'] * np.log1p(df['total_receipts_amount'])
    
    # Binary indicators
    features['is_high_miles'] = (df['miles_traveled'] > 1000).astype(int)
    features['is_low_receipts'] = (df['total_receipts_amount'] < 50).astype(int)
    features['is_49_ending'] = ((df['total_receipts_amount'] * 100).astype(int) % 100 == 49).astype(int)
    features['is_99_ending'] = ((df['total_receipts_amount'] * 100).astype(int) % 100 == 99).astype(int)
    
    return features

# Improved cluster assignment with merging
def assign_cluster_merged(row):
    """Assign cluster with small cluster merging"""
    days = row['trip_days']
    miles = row['miles_traveled']
    receipts = row['total_receipts_amount']
    
    # First get original cluster
    if days == 1:
        if miles >= 600:
            return '1_high_miles'  # Merge 1a and 1b
        else:
            return '6'
    elif days < 100 and miles < 200 and receipts > 1500:
        return '0'  # Merge with main cluster 0 (too few samples)
    elif days >= 10 and receipts >= 1100:
        return '2'
    elif 3 <= days <= 5 and receipts >= 1400:
        return '3'
    elif receipts <= 24:
        return '0'  # Merge cluster 4 with 0 (too few samples)
    elif 5 <= days <= 12 and miles >= 700:
        return '5'  # Merge 5_special with 5
    else:
        return '0'

# Generate predictions and residuals
print("\nGenerating rule engine predictions...")
for df_name, df in [('train', train_main), ('blind', train_blind)]:
    predictions = []
    for _, row in df.iterrows():
        pred = predict_reimbursement(
            row['trip_days'],
            row['miles_traveled'],
            row['total_receipts_amount']
        )
        predictions.append(pred)
    df['rule_engine_pred'] = predictions
    df['residual'] = df['expected_reimbursement'] - df['rule_engine_pred']
    df['cluster'] = df.apply(assign_cluster_merged, axis=1)

# Improved per-cluster ensemble
class RobustPerClusterEnsemble:
    def __init__(self):
        self.cluster_models = {}
        self.cluster_bounds = {}
        self.cluster_baselines = {}
        self.cluster_stats = {}
        self.global_fallback_mae = None
        
    def fit(self, X, y, clusters):
        """Train separate models for each cluster with improved handling"""
        unique_clusters = sorted(clusters.unique())
        
        # First compute global fallback (for very small clusters)
        self.global_fallback_mae = y.mean()
        print(f"\nGlobal fallback baseline: ${self.global_fallback_mae:.2f}")
        
        for cluster_id in unique_clusters:
            print(f"\nTraining cluster {cluster_id}...")
            mask = clusters == cluster_id
            X_cluster = X[mask]
            y_cluster = y[mask]
            
            n_samples = len(y_cluster)
            print(f"  Samples: {n_samples}")
            
            # Compute baseline
            baseline = y_cluster.mean()
            residual_std = y_cluster.std()
            
            print(f"  Baseline residual: ${baseline:.2f}")
            print(f"  Residual std: ${residual_std:.2f}")
            
            # Minimum samples threshold
            min_samples_for_model = 30
            
            if n_samples < min_samples_for_model:
                # Use conservative approach for small clusters
                print(f"  Using conservative baseline only (n < {min_samples_for_model})")
                
                # Use weighted average with global baseline
                weight = n_samples / min_samples_for_model
                weighted_baseline = weight * baseline + (1 - weight) * self.global_fallback_mae
                
                self.cluster_baselines[cluster_id] = weighted_baseline
                self.cluster_bounds[cluster_id] = min(residual_std * 0.5, 50)  # Very conservative
                self.cluster_models[cluster_id] = None
                self.cluster_stats[cluster_id] = {
                    'n_samples': n_samples,
                    'baseline': float(weighted_baseline),
                    'bound': float(self.cluster_bounds[cluster_id]),
                    'confidence': weight * 0.5  # Low confidence
                }
                continue
            
            # For larger clusters, train models
            y_demeaned = y_cluster - baseline
            
            # Adaptive model complexity
            if n_samples < 100:
                max_depth = 4
                n_estimators = 50
                min_samples_leaf = max(5, n_samples // 20)
            else:
                max_depth = 6
                n_estimators = 100
                min_samples_leaf = 5
            
            models = {
                'ExtraTrees': ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=10,
                    min_samples_leaf=min_samples_leaf,
                    max_features='sqrt',  # More regularization
                    random_state=42,
                    n_jobs=-1
                ),
                'GBM': GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=0.03,  # Lower learning rate
                    max_depth=max_depth,
                    min_samples_split=10,
                    min_samples_leaf=min_samples_leaf,
                    subsample=0.7,
                    max_features='sqrt',
                    random_state=42
                )
            }
            
            # Train models
            predictions = []
            weights = {'ExtraTrees': 0.5, 'GBM': 0.5}  # Simpler weighting
            
            for name, model in models.items():
                model.fit(X_cluster, y_demeaned)
                pred = model.predict(X_cluster)
                predictions.append(pred * weights[name])
            
            combined_pred = np.sum(predictions, axis=0)
            
            # Compute conservative bound
            errors = np.abs(combined_pred)
            
            # Use 90th percentile for more conservative bounds
            base_bound = np.percentile(errors, 90)
            
            # Scale bound based on cluster size and variability
            size_factor = min(1.0, n_samples / 200)  # Full confidence at 200+ samples
            variability_factor = min(1.0, residual_std / 100)  # Scale by residual magnitude
            
            final_bound = base_bound * (0.5 + 0.5 * size_factor) * (0.7 + 0.3 * variability_factor)
            
            print(f"  90th percentile error: ${base_bound:.2f}")
            print(f"  Adjusted bound: ${final_bound:.2f}")
            
            # Confidence based on sample size and fit quality
            train_mae = mean_absolute_error(y_demeaned, combined_pred)
            fit_quality = 1 - min(1.0, train_mae / residual_std)  # How well we fit vs baseline std
            confidence = min(1.0, size_factor * fit_quality)
            
            print(f"  Confidence: {confidence:.2f}")
            
            # Store results
            self.cluster_models[cluster_id] = models
            self.cluster_bounds[cluster_id] = final_bound
            self.cluster_baselines[cluster_id] = baseline
            self.cluster_stats[cluster_id] = {
                'n_samples': n_samples,
                'baseline': float(baseline),
                'bound': float(final_bound),
                'confidence': float(confidence),
                'train_mae': float(train_mae)
            }
    
    def predict(self, X, clusters):
        """Predict with improved robustness"""
        predictions = np.zeros(len(X))
        
        for cluster_id in self.cluster_baselines:
            mask = clusters == cluster_id
            if not mask.any():
                continue
                
            X_cluster = X[mask]
            baseline = self.cluster_baselines[cluster_id]
            
            if self.cluster_models[cluster_id] is None:
                # No model - use baseline
                predictions[mask] = baseline
            else:
                # Get predictions
                cluster_preds = []
                weights = {'ExtraTrees': 0.5, 'GBM': 0.5}
                
                for name, model in self.cluster_models[cluster_id].items():
                    if name in weights:
                        pred = model.predict(X_cluster)
                        cluster_preds.append(pred * weights[name])
                
                demeaned_pred = np.sum(cluster_preds, axis=0)
                
                # Apply bound
                bound = self.cluster_bounds[cluster_id]
                bounded_pred = np.clip(demeaned_pred, -bound, bound)
                
                # Apply confidence
                confidence = self.cluster_stats[cluster_id]['confidence']
                final_correction = baseline + confidence * bounded_pred
                
                predictions[mask] = final_correction
        
        return predictions

# Train improved ensemble
print("\n=== Training Robust Per-Cluster Ensemble ===")

# Prepare features
X_train = engineer_features(train_main)
y_train = train_main['residual']
clusters_train = train_main['cluster']

X_blind = engineer_features(train_blind)
y_blind = train_blind['residual']
clusters_blind = train_blind['cluster']

# Show cluster distribution
print("\nCluster distribution in training set:")
print(clusters_train.value_counts().sort_index())

# Train
ensemble = RobustPerClusterEnsemble()
ensemble.fit(X_train, y_train, clusters_train)

# Evaluate on training
print("\n=== TRAINING SET PERFORMANCE ===")
train_corrections = ensemble.predict(X_train, clusters_train)
train_final_pred = train_main['rule_engine_pred'] + train_corrections
train_mae = mean_absolute_error(train_main['expected_reimbursement'], train_final_pred)
train_max_error = abs(train_main['expected_reimbursement'] - train_final_pred).max()

print(f"\nOverall Training MAE: ${train_mae:.2f}")
print(f"Max Training Error: ${train_max_error:.2f}")

# Evaluate on blind
print("\n=== BLIND TEST SET PERFORMANCE ===")
blind_corrections = ensemble.predict(X_blind, clusters_blind)
blind_final_pred = train_blind['rule_engine_pred'] + blind_corrections
blind_mae = mean_absolute_error(train_blind['expected_reimbursement'], blind_final_pred)
blind_max_error = abs(train_blind['expected_reimbursement'] - blind_final_pred).max()

print(f"\nOverall Blind Test MAE: ${blind_mae:.2f}")
print(f"Max Blind Test Error: ${blind_max_error:.2f}")

# Per-cluster blind performance
print("\nPer-cluster blind test performance:")
blind_results = train_blind.copy()
blind_results['final_pred'] = blind_final_pred
blind_results['error'] = abs(blind_results['expected_reimbursement'] - blind_results['final_pred'])

cluster_maes = {}
for cluster in sorted(blind_results['cluster'].unique()):
    cluster_data = blind_results[blind_results['cluster'] == cluster]
    if len(cluster_data) > 0:
        mae = cluster_data['error'].mean()
        max_err = cluster_data['error'].max()
        count = len(cluster_data)
        cluster_maes[cluster] = mae
        print(f"  Cluster {cluster}: MAE ${mae:.2f}, Max ${max_err:.2f} ({count} samples)")

max_cluster_mae = max(cluster_maes.values()) if cluster_maes else 0

# Success check
print("\n=== SUCCESS CRITERIA CHECK ===")
if blind_mae <= 40:
    print(f"✅ Blind MAE ${blind_mae:.2f} <= $40")
else:
    print(f"❌ Blind MAE ${blind_mae:.2f} > $40")

if max_cluster_mae <= 70:
    print(f"✅ Max cluster MAE ${max_cluster_mae:.2f} <= $70")
else:
    print(f"❌ Max cluster MAE ${max_cluster_mae:.2f} > $70")

# Save if improved
if blind_mae < 50:  # Relaxed criteria for saving
    print("\n💾 Saving improved models...")
    joblib.dump(ensemble, 'robust_per_cluster_ensemble.pkl')
    
    config = {
        'cluster_stats': ensemble.cluster_stats,
        'training_mae': float(train_mae),
        'blind_mae': float(blind_mae),
        'max_cluster_mae': float(max_cluster_mae),
        'cluster_mapping': 'merged small clusters'
    }
    
    with open('robust_per_cluster_config.json', 'w') as f:
        json.dump(config, f, indent=2)

# Analysis of improvements
print("\n=== IMPROVEMENT ANALYSIS ===")
print("Changes from v1:")
print("- Merged small clusters (1a+1b→1_high_miles, 4→0, etc.)")
print("- More conservative bounds (90th percentile)")
print("- Adaptive model complexity based on cluster size")
print("- Confidence weighting based on sample size and fit quality")
print("- Reduced feature complexity for stability")

print("\n" + "="*60)
print("Training complete!") 