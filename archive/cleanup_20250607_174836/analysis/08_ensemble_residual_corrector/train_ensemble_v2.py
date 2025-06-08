#!/usr/bin/env python3
"""
Train ensemble v2 with robustness improvements:
1. Blind holdout set (50 rows) for overfitting detection
2. Winsorized corrections (±$75 cap)
3. Confidence gating using prediction variance
4. Cluster-wise performance analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Create blind holdout set (50 rows)
print("\nCreating blind holdout set...")
train_main, train_blind = train_test_split(train_df, test_size=50, random_state=42, stratify=None)
print(f"Main training set: {len(train_main)} rows")
print(f"Blind holdout set: {len(train_blind)} rows")

# Load v3 model
print("\nLoading v3 rule engine...")
import sys
sys.path.append('.')
from models.cluster_models_optimized import calculate_reimbursement_v3 as predict_reimbursement

# Generate base predictions
print("\nGenerating rule engine predictions...")
for df_name, df in [('main', train_main), ('blind', train_blind), ('full', train_df)]:
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

# Feature engineering
print("\nEngineering features...")
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
    
    # Binned features
    features['days_bin'] = pd.cut(df['trip_days'], bins=[0, 1, 3, 5, 10, 20], labels=False)
    features['miles_bin'] = pd.cut(df['miles_traveled'], bins=[0, 100, 500, 1000, 2000, 5000], labels=False)
    features['receipts_bin'] = pd.cut(df['total_receipts_amount'], bins=[0, 50, 500, 1000, 2000, 5000], labels=False)
    
    # Rule engine error magnitude
    features['pred_magnitude'] = abs(df['rule_engine_pred'])
    features['pred_log'] = np.log1p(abs(df['rule_engine_pred']))
    
    return features

X_train_main = engineer_features(train_main)
y_train_main = train_main['residual']

X_train_blind = engineer_features(train_blind)
y_train_blind = train_blind['residual']

# Define models with modifications for uncertainty estimation
print("\nDefining ensemble models...")
models = {
    'ExtraTrees': ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,  # Enable for OOB scores
        oob_score=True
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        oob_score=True  # Already uses bootstrap by default
    ),
    'GBM': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,  # Use for early stopping
        n_iter_no_change=10
    )
}

# Train on main set
print("\nTraining models on main set (excluding blind holdout)...")
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_main, y_train_main)
    trained_models[name] = model
    
    # Evaluate on blind holdout
    blind_pred = model.predict(X_train_blind)
    blind_mae = mean_absolute_error(y_train_blind, blind_pred)
    print(f"  Blind holdout MAE: ${blind_mae:.2f}")

# Function to predict with uncertainty
def predict_with_trees_uncertainty(models_dict, X):
    """Get predictions and uncertainty estimates from tree models"""
    predictions = {}
    uncertainties = {}
    
    for name, model in models_dict.items():
        if name in ['ExtraTrees', 'RandomForest']:
            # Get predictions from all trees
            tree_preds = []
            for tree in model.estimators_:
                tree_pred = tree.predict(X)
                tree_preds.append(tree_pred)
            tree_preds = np.array(tree_preds)
            
            # Mean and std across trees
            predictions[name] = tree_preds.mean(axis=0)
            uncertainties[name] = tree_preds.std(axis=0)
        else:
            # GBM - use staged predictions for uncertainty proxy
            predictions[name] = model.predict(X)
            # Simple heuristic: use prediction magnitude as uncertainty proxy
            uncertainties[name] = np.abs(predictions[name]) * 0.1
    
    return predictions, uncertainties

# Enhanced prediction function with winsorization and confidence gating
def predict_with_ensemble_v2(trip_days, miles_traveled, total_receipts_amount, models_dict, 
                            winsorize_limit=75, confidence_threshold=35):
    """
    Predict using rule engine + ensemble correction with:
    1. Winsorized corrections (capped at ±winsorize_limit)
    2. Confidence gating (fall back to rule engine if uncertainty > threshold)
    """
    # Get base prediction from rule engine
    base_pred = predict_reimbursement(trip_days, miles_traveled, total_receipts_amount)
    
    # Create feature vector
    df_single = pd.DataFrame({
        'trip_days': [trip_days],
        'miles_traveled': [miles_traveled],
        'total_receipts_amount': [total_receipts_amount],
        'rule_engine_pred': [base_pred]
    })
    
    features = engineer_features(df_single)
    
    # Get predictions and uncertainties
    preds, uncerts = predict_with_trees_uncertainty(models_dict, features)
    
    # Weighted ensemble
    weights = {'ExtraTrees': 0.4, 'GBM': 0.35, 'RandomForest': 0.25}
    
    # Calculate weighted correction and uncertainty
    weighted_correction = 0
    weighted_uncertainty = 0
    for name in weights:
        weighted_correction += preds[name][0] * weights[name]
        weighted_uncertainty += uncerts[name][0] * weights[name]
    
    # Winsorize the correction
    correction_raw = weighted_correction
    correction_winsorized = np.clip(correction_raw, -winsorize_limit, winsorize_limit)
    
    # Confidence gating - if uncertainty too high, reduce correction
    if weighted_uncertainty > confidence_threshold:
        # High uncertainty - trust rule engine more
        confidence_factor = confidence_threshold / weighted_uncertainty
        correction_gated = correction_winsorized * confidence_factor
    else:
        correction_gated = correction_winsorized
    
    # Final prediction
    final_pred = base_pred + correction_gated
    
    # Ensure non-negative
    final_pred = max(0, final_pred)
    
    return {
        'final_prediction': round(final_pred, 2),
        'base_prediction': base_pred,
        'correction_raw': correction_raw,
        'correction_winsorized': correction_winsorized,
        'correction_final': correction_gated,
        'uncertainty': weighted_uncertainty,
        'confidence_gated': weighted_uncertainty > confidence_threshold
    }

# Evaluate on blind holdout
print("\n" + "="*80)
print("EVALUATING ON BLIND HOLDOUT SET")
print("="*80)

blind_results = []
for _, row in train_blind.iterrows():
    result = predict_with_ensemble_v2(
        row['trip_days'],
        row['miles_traveled'],
        row['total_receipts_amount'],
        trained_models
    )
    result['expected'] = row['expected_reimbursement']
    blind_results.append(result)

blind_df = pd.DataFrame(blind_results)
blind_df['error'] = blind_df['final_prediction'] - blind_df['expected']
blind_df['abs_error'] = abs(blind_df['error'])

# Blind holdout metrics
print(f"\nBlind Holdout Results:")
print(f"MAE: ${blind_df['abs_error'].mean():.2f}")
print(f"Max Error: ${blind_df['abs_error'].max():.2f}")
print(f"Cases with confidence gating: {blind_df['confidence_gated'].sum()}")

# Compare different correction strategies
print(f"\nCorrection Analysis:")
print(f"Average raw correction: ${abs(blind_df['correction_raw']).mean():.2f}")
print(f"Average winsorized correction: ${abs(blind_df['correction_winsorized']).mean():.2f}")
print(f"Average final correction: ${abs(blind_df['correction_final']).mean():.2f}")
print(f"Cases clipped by winsorization: {(abs(blind_df['correction_raw']) > 75).sum()}")

# Evaluate on full training set with new approach
print("\n" + "="*80)
print("EVALUATING ON FULL TRAINING SET")
print("="*80)

# Retrain on full data for final model
print("\nRetraining on full dataset for final model...")
X_train_full = engineer_features(train_df)
y_train_full = train_df['residual']

final_models = {}
for name, model in models.items():
    print(f"Training final {name}...")
    model.fit(X_train_full, y_train_full)
    final_models[name] = model

# Evaluate with different strategies
strategies = {
    'v1_unconstrained': {'winsorize': 1000, 'confidence': 1000},  # Original
    'v2_winsorized': {'winsorize': 75, 'confidence': 1000},       # Just winsorization
    'v2_gated': {'winsorize': 1000, 'confidence': 35},            # Just gating
    'v2_full': {'winsorize': 75, 'confidence': 35}                # Both
}

results_by_strategy = {}
for strategy_name, params in strategies.items():
    print(f"\nTesting strategy: {strategy_name}")
    strategy_results = []
    
    for _, row in train_df.iterrows():
        result = predict_with_ensemble_v2(
            row['trip_days'],
            row['miles_traveled'],
            row['total_receipts_amount'],
            final_models,
            winsorize_limit=params['winsorize'],
            confidence_threshold=params['confidence']
        )
        result['expected'] = row['expected_reimbursement']
        strategy_results.append(result)
    
    strategy_df = pd.DataFrame(strategy_results)
    strategy_df['error'] = strategy_df['final_prediction'] - strategy_df['expected']
    strategy_df['abs_error'] = abs(strategy_df['error'])
    
    mae = strategy_df['abs_error'].mean()
    max_error = strategy_df['abs_error'].max()
    gated_count = strategy_df['confidence_gated'].sum() if 'confidence_gated' in strategy_df else 0
    
    print(f"  MAE: ${mae:.2f}")
    print(f"  Max Error: ${max_error:.2f}")
    print(f"  Gated cases: {gated_count}")
    
    results_by_strategy[strategy_name] = strategy_df

# Cluster analysis for best strategy
print("\n" + "="*80)
print("CLUSTER ANALYSIS FOR v2_full STRATEGY")
print("="*80)

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

# Add cluster assignments
best_df = results_by_strategy['v2_full'].copy()
train_df_with_clusters = train_df.copy()
train_df_with_clusters['cluster'] = train_df_with_clusters.apply(assign_cluster, axis=1)

# Merge cluster info
best_df['cluster'] = train_df_with_clusters['cluster']

# Analyze by cluster
cluster_stats = best_df.groupby('cluster').agg({
    'abs_error': ['count', 'mean', 'max'],
    'confidence_gated': 'sum',
    'correction_final': lambda x: abs(x).mean()
}).round(2)

print("\nPerformance by cluster:")
print(cluster_stats)

# Save the v2 models
print("\n" + "="*80)
print("SAVING FINAL MODELS")
print("="*80)

for name, model in final_models.items():
    filename = f'analysis/08_ensemble_residual_corrector/{name.lower()}_residual_model_v2.pkl'
    joblib.dump(model, filename)
    print(f"Saved {name} to {filename}")

# Save configuration
config = {
    'winsorize_limit': 75,
    'confidence_threshold': 35,
    'weights': {'ExtraTrees': 0.4, 'GBM': 0.35, 'RandomForest': 0.25}
}
import json
with open('analysis/08_ensemble_residual_corrector/ensemble_v2_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\nTraining complete! Use v2 models for more robust predictions.") 