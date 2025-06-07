#!/usr/bin/env python3
"""
Per-Cluster Ensemble Training - Following Sprint A & B Plan

Sprint A: Localize & tame
- Train one residual model per cluster
- Compute 95th percentile bounds per cluster  
- Add per-cluster mean residual as baseline

Sprint B: True blind test
- 150/850 split (15% test)
- Target: Blind MAE < $40, no cluster > $70
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('../../data/train.csv')

# Sprint B: Create proper 15% blind test set
print("\n=== SPRINT B: Creating 15% blind test set ===")
train_main, train_blind = train_test_split(train_df, test_size=0.15, random_state=42)
print(f"Training set: {len(train_main)} rows")
print(f"Blind test set: {len(train_blind)} rows")

# Load v3 rule engine
print("\nLoading v3 rule engine...")
import sys
sys.path.append('../..')
from models.cluster_models_optimized import calculate_reimbursement_v3 as predict_reimbursement

# Feature engineering function
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
    
    return features

# Cluster assignment function
def assign_cluster(row):
    """Assign cluster based on v3 rules"""
    days = row['trip_days']
    miles = row['miles_traveled']
    receipts = row['total_receipts_amount']
    
    if days == 1:
        if miles >= 600:
            if receipts >= 800:
                return '1a'
            else:
                return '1b'
        else:
            return '6'
    elif days < 100 and miles < 200 and receipts > 1500:
        return '0_low_mile_high_receipt'
    elif days >= 10 and receipts >= 1100:
        return '2'
    elif 3 <= days <= 5 and receipts >= 1400:
        return '3'
    elif receipts <= 24:
        return '4'
    elif 5 <= days <= 12 and miles >= 700:
        if 7 <= days <= 8 and 900 <= miles <= 1200 and 1000 <= receipts <= 1200:
            return '5_special'
        else:
            return '5'
    else:
        return '0'

# Generate predictions and residuals for both sets
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
    df['cluster'] = df.apply(assign_cluster, axis=1)

# Sprint A: Per-cluster ensemble
print("\n=== SPRINT A: Training per-cluster ensembles ===")

class PerClusterEnsemble:
    def __init__(self):
        self.cluster_models = {}
        self.cluster_bounds = {}
        self.cluster_baselines = {}
        self.cluster_stats = {}
        
    def fit(self, X, y, clusters):
        """Train separate models for each cluster"""
        unique_clusters = sorted(clusters.unique())
        
        for cluster_id in unique_clusters:
            print(f"\nTraining cluster {cluster_id}...")
            mask = clusters == cluster_id
            X_cluster = X[mask]
            y_cluster = y[mask]
            
            n_samples = len(y_cluster)
            print(f"  Samples: {n_samples}")
            
            if n_samples < 10:
                # Too few samples - use mean only
                self.cluster_baselines[cluster_id] = y_cluster.mean()
                self.cluster_bounds[cluster_id] = 50  # Conservative bound
                self.cluster_models[cluster_id] = None
                self.cluster_stats[cluster_id] = {
                    'n_samples': n_samples,
                    'baseline': float(y_cluster.mean()),
                    'bound': 50,
                    'confidence': 0.1
                }
                continue
            
            # A3: Compute baseline (mean residual)
            baseline = y_cluster.mean()
            y_demeaned = y_cluster - baseline
            
            print(f"  Baseline residual: ${baseline:.2f}")
            print(f"  Residual std: ${y_demeaned.std():.2f}")
            
            # A1: Train ensemble (with reduced complexity for small clusters)
            max_depth = min(6, int(np.log2(n_samples)))
            min_samples_leaf = max(2, n_samples // 50)
            
            models = {
                'ExtraTrees': ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=max_depth,
                    min_samples_split=5,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                ),
                'RandomForest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=max_depth,
                    min_samples_split=5,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                ),
                'GBM': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=max_depth,
                    min_samples_split=10,
                    min_samples_leaf=min_samples_leaf,
                    subsample=0.8,
                    random_state=42
                )
            }
            
            # Train and combine models
            predictions = []
            weights = {'ExtraTrees': 0.4, 'RandomForest': 0.25, 'GBM': 0.35}
            
            for name, model in models.items():
                model.fit(X_cluster, y_demeaned)
                pred = model.predict(X_cluster)
                predictions.append(pred * weights[name])
            
            combined_pred = np.sum(predictions, axis=0)
            
            # A2: Compute 95th percentile bound
            errors = np.abs(combined_pred)
            bound = np.percentile(errors, 95)
            
            print(f"  95th percentile bound: ${bound:.2f}")
            
            # Confidence based on sample size
            confidence = min(1.0, n_samples / 100)
            
            # Store results
            self.cluster_models[cluster_id] = models
            self.cluster_bounds[cluster_id] = bound
            self.cluster_baselines[cluster_id] = baseline
            self.cluster_stats[cluster_id] = {
                'n_samples': n_samples,
                'baseline': float(baseline),
                'bound': float(bound),
                'confidence': float(confidence)
            }
    
    def predict(self, X, clusters):
        """Predict using per-cluster models"""
        predictions = np.zeros(len(X))
        
        for cluster_id in self.cluster_models:
            mask = clusters == cluster_id
            if not mask.any():
                continue
                
            X_cluster = X[mask]
            
            # Get baseline
            baseline = self.cluster_baselines[cluster_id]
            
            if self.cluster_models[cluster_id] is None:
                # No model - use baseline only
                predictions[mask] = baseline
            else:
                # Get predictions from each model
                cluster_preds = []
                weights = {'ExtraTrees': 0.4, 'RandomForest': 0.25, 'GBM': 0.35}
                
                for name, model in self.cluster_models[cluster_id].items():
                    pred = model.predict(X_cluster)
                    cluster_preds.append(pred * weights[name])
                
                # Combine and add baseline
                demeaned_pred = np.sum(cluster_preds, axis=0)
                
                # Apply bound
                bound = self.cluster_bounds[cluster_id]
                bounded_pred = np.clip(demeaned_pred, -bound, bound)
                
                # Apply confidence
                confidence = self.cluster_stats[cluster_id]['confidence']
                final_correction = baseline + confidence * bounded_pred
                
                predictions[mask] = final_correction
        
        return predictions

# Prepare features
print("\nPreparing features...")
X_train = engineer_features(train_main)
y_train = train_main['residual']
clusters_train = train_main['cluster']

X_blind = engineer_features(train_blind)
y_blind = train_blind['residual']
clusters_blind = train_blind['cluster']

# Train per-cluster ensemble
ensemble = PerClusterEnsemble()
ensemble.fit(X_train, y_train, clusters_train)

# Evaluate on training set
print("\n=== EVALUATING ON TRAINING SET ===")
train_corrections = ensemble.predict(X_train, clusters_train)
train_final_pred = train_main['rule_engine_pred'] + train_corrections
train_mae = mean_absolute_error(train_main['expected_reimbursement'], train_final_pred)
train_max_error = abs(train_main['expected_reimbursement'] - train_final_pred).max()

print(f"\nOverall Training MAE: ${train_mae:.2f}")
print(f"Max Training Error: ${train_max_error:.2f}")

# Per-cluster training performance
print("\nPer-cluster training performance:")
train_results = train_main.copy()
train_results['final_pred'] = train_final_pred
train_results['error'] = abs(train_results['expected_reimbursement'] - train_results['final_pred'])

for cluster in sorted(train_results['cluster'].unique()):
    cluster_data = train_results[train_results['cluster'] == cluster]
    mae = cluster_data['error'].mean()
    max_err = cluster_data['error'].max()
    count = len(cluster_data)
    print(f"  Cluster {cluster}: MAE ${mae:.2f}, Max ${max_err:.2f} ({count} samples)")

# Evaluate on blind test set
print("\n=== EVALUATING ON BLIND TEST SET ===")
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

for cluster in sorted(blind_results['cluster'].unique()):
    cluster_data = blind_results[blind_results['cluster'] == cluster]
    if len(cluster_data) > 0:
        mae = cluster_data['error'].mean()
        max_err = cluster_data['error'].max()
        count = len(cluster_data)
        print(f"  Cluster {cluster}: MAE ${mae:.2f}, Max ${max_err:.2f} ({count} samples)")

# Check success criteria
print("\n=== SUCCESS CRITERIA CHECK ===")
success = True

if blind_mae <= 40:
    print(f"✅ Blind MAE ${blind_mae:.2f} <= $40")
else:
    print(f"❌ Blind MAE ${blind_mae:.2f} > $40")
    success = False

# Check per-cluster criteria
max_cluster_mae = 0
for cluster in sorted(blind_results['cluster'].unique()):
    cluster_data = blind_results[blind_results['cluster'] == cluster]
    if len(cluster_data) > 0:
        mae = cluster_data['error'].mean()
        max_cluster_mae = max(max_cluster_mae, mae)

if max_cluster_mae <= 70:
    print(f"✅ Max cluster MAE ${max_cluster_mae:.2f} <= $70")
else:
    print(f"❌ Max cluster MAE ${max_cluster_mae:.2f} > $70")
    success = False

# Save models if successful
if success:
    print("\n✅ SUCCESS! Saving models...")
    
    # Save ensemble
    joblib.dump(ensemble, 'per_cluster_ensemble.pkl')
    
    # Save configuration
    config = {
        'cluster_stats': ensemble.cluster_stats,
        'training_mae': float(train_mae),
        'blind_mae': float(blind_mae),
        'max_cluster_mae': float(max_cluster_mae)
    }
    
    with open('per_cluster_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Models saved successfully!")
else:
    print("\n❌ Did not meet success criteria. Consider tuning hyperparameters.")

# Show worst blind test cases
print("\n=== WORST BLIND TEST CASES ===")
worst_cases = blind_results.nlargest(10, 'error')[
    ['trip_days', 'miles_traveled', 'total_receipts_amount', 
     'expected_reimbursement', 'final_pred', 'error', 'cluster']
]
print(worst_cases.to_string(index=False))

print("\n" + "="*60)
print("Training complete!") 