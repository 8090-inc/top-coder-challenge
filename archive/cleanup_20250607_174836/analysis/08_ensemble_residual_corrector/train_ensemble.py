#!/usr/bin/env python3
"""
Train a stacked ensemble model to correct rule engine residuals.

This script:
1. Uses the v3 rule engine as the base predictor
2. Engineers features from the input data
3. Trains ExtraTrees, GBM, and RF to predict residuals
4. Combines predictions for improved accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Load v3 model
print("Loading v3 rule engine...")
import sys
sys.path.append('.')
from models.cluster_models_optimized import calculate_reimbursement_v3 as predict_reimbursement

# Generate base predictions from rule engine
print("Generating rule engine predictions...")
train_predictions = []
for _, row in train_df.iterrows():
    pred = predict_reimbursement(
        row['trip_days'],
        row['miles_traveled'],
        row['total_receipts_amount']
    )
    train_predictions.append(pred)

train_df['rule_engine_pred'] = train_predictions
train_df['residual'] = train_df['expected_reimbursement'] - train_df['rule_engine_pred']

# Feature engineering
print("Engineering features...")
def engineer_features(df):
    features = pd.DataFrame()
    
    # Basic features
    features['trip_days'] = df['trip_days']
    features['miles_traveled'] = df['miles_traveled']
    features['total_receipts_amount'] = df['total_receipts_amount']
    
    # Rule engine prediction as feature
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
    
    # Rule engine error magnitude (helps learn correction patterns)
    features['pred_magnitude'] = abs(df['rule_engine_pred'])
    features['pred_log'] = np.log1p(abs(df['rule_engine_pred']))
    
    return features

X_train = engineer_features(train_df)
y_train = train_df['residual']  # Target is the residual to correct

print(f"\nFeature set shape: {X_train.shape}")
print(f"Target (residual) stats: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

# Define models
print("\nDefining ensemble models...")
models = {
    'ExtraTrees': ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'GBM': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# Cross-validation
print("\nPerforming cross-validation...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Evaluate residual prediction
    residual_scores = []
    final_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train on residuals
        model.fit(X_fold_train, y_fold_train)
        
        # Predict residuals
        residual_pred = model.predict(X_fold_val)
        
        # Calculate residual prediction MAE
        residual_mae = mean_absolute_error(y_fold_val, residual_pred)
        residual_scores.append(residual_mae)
        
        # Calculate final prediction MAE
        final_pred = X_fold_val['rule_engine_pred'] + residual_pred
        actual = train_df.iloc[val_idx]['expected_reimbursement']
        final_mae = mean_absolute_error(actual, final_pred)
        final_scores.append(final_mae)
        
        print(f"  Fold {fold+1}: Residual MAE=${residual_mae:.2f}, Final MAE=${final_mae:.2f}")
    
    cv_results[name] = {
        'residual_mae': np.mean(residual_scores),
        'final_mae': np.mean(final_scores)
    }
    print(f"  Average - Residual MAE: ${np.mean(residual_scores):.2f}, Final MAE: ${np.mean(final_scores):.2f}")

# Train final models on full data
print("\nTraining final models on full dataset...")
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nTop 10 features for {name}:")
        print(importances.head(10))

# Create stacked ensemble predictor
print("\nCreating stacked ensemble...")
def predict_with_ensemble(trip_days, miles_traveled, total_receipts_amount, models_dict):
    """Predict using rule engine + ensemble correction."""
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
    
    # Get residual predictions from each model
    residual_preds = []
    weights = {'ExtraTrees': 0.4, 'GBM': 0.35, 'RandomForest': 0.25}  # Based on CV performance
    
    for name, model in models_dict.items():
        pred = model.predict(features)[0]
        residual_preds.append(pred * weights[name])
    
    # Weighted average of residual predictions
    residual_correction = sum(residual_preds)
    
    # Final prediction
    final_pred = base_pred + residual_correction
    
    return final_pred, base_pred, residual_correction

# Test on training data
print("\nEvaluating final ensemble on training data...")
ensemble_predictions = []
base_predictions = []
corrections = []

for _, row in train_df.iterrows():
    final_pred, base_pred, correction = predict_with_ensemble(
        row['trip_days'],
        row['miles_traveled'],
        row['total_receipts_amount'],
        trained_models
    )
    ensemble_predictions.append(final_pred)
    base_predictions.append(base_pred)
    corrections.append(correction)

train_df['ensemble_pred'] = ensemble_predictions
train_df['correction'] = corrections

# Calculate metrics
base_mae = mean_absolute_error(train_df['expected_reimbursement'], base_predictions)
ensemble_mae = mean_absolute_error(train_df['expected_reimbursement'], ensemble_predictions)
improvement = (base_mae - ensemble_mae) / base_mae * 100

print(f"\nResults:")
print(f"Base Rule Engine MAE: ${base_mae:.2f}")
print(f"Ensemble Corrected MAE: ${ensemble_mae:.2f}")
print(f"Improvement: {improvement:.1f}%")

# Error distribution
errors = abs(train_df['expected_reimbursement'] - train_df['ensemble_pred'])
print(f"\nError Distribution:")
print(f"Max Error: ${errors.max():.2f}")
print(f"Errors > $500: {(errors > 500).sum()}")
print(f"Errors > $300: {(errors > 300).sum()}")
print(f"90th percentile: ${np.percentile(errors, 90):.2f}")

# Save models
print("\nSaving models...")
for name, model in trained_models.items():
    joblib.dump(model, f'analysis/08_ensemble_residual_corrector/{name.lower()}_residual_model.pkl')

# Generate test predictions
print("\nGenerating test predictions...")
test_predictions = []
test_base_predictions = []
test_corrections = []

for _, row in test_df.iterrows():
    final_pred, base_pred, correction = predict_with_ensemble(
        row['trip_days'],
        row['miles_traveled'],
        row['total_receipts_amount'],
        trained_models
    )
    test_predictions.append(final_pred)
    test_base_predictions.append(base_pred)
    test_corrections.append(correction)

# Save predictions
output_df = pd.DataFrame({
    'trip_days': test_df['trip_days'],
    'miles_traveled': test_df['miles_traveled'],
    'total_receipts_amount': test_df['total_receipts_amount'],
    'base_prediction': test_base_predictions,
    'correction': test_corrections,
    'final_prediction': test_predictions
})

output_df.to_csv('analysis/08_ensemble_residual_corrector/ensemble_test_predictions.csv', index=False)
print(f"\nTest predictions saved to analysis/08_ensemble_residual_corrector/ensemble_test_predictions.csv")

# Analyze corrections by cluster
print("\nAnalyzing corrections by cluster...")
# Assign clusters to understand where corrections help most
def assign_cluster(row):
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

train_df['cluster'] = train_df.apply(assign_cluster, axis=1)

cluster_stats = train_df.groupby('cluster').agg({
    'correction': ['mean', 'std', 'count'],
    'residual': 'mean'
}).round(2)

print("\nCorrection statistics by cluster:")
print(cluster_stats)

print("\nEnsemble training complete!") 