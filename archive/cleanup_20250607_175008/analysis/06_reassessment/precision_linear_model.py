"""
Precision-Focused Linear Model Development
Goal: Get exact coefficients that produce the right decimal patterns
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

print("=" * 80)
print("PRECISION LINEAR MODEL DEVELOPMENT")
print("=" * 80)

# Load data
with open(PUBLIC_CASES_PATH, "r") as f:
    import json
    data = json.load(f)

df = pd.json_normalize(data)
df.columns = ['expected_output', 'trip_days', 'miles', 'receipts']

# Add derived features that might help with precision
df['miles_per_day'] = df['miles'] / df['trip_days']
df['receipts_per_day'] = df['receipts'] / df['trip_days']
df['trip_days_squared'] = df['trip_days'] ** 2
df['miles_squared'] = df['miles'] ** 2
df['receipts_squared'] = df['receipts'] ** 2
df['sqrt_miles'] = np.sqrt(df['miles'])
df['sqrt_receipts'] = np.sqrt(df['receipts'])
df['log_receipts'] = np.log1p(df['receipts'])  # log(1 + receipts) to handle small values

# Check for receipt endings
df['receipt_cents'] = (df['receipts'] * 100) % 100
df['has_49'] = (df['receipt_cents'] == 49).astype(int)
df['has_99'] = (df['receipt_cents'] == 99).astype(int)

print(f"Loaded {len(df)} cases")

# Test different feature combinations
print("\n" + "=" * 40)
print("TESTING FEATURE COMBINATIONS")
print("=" * 40)

feature_sets = {
    "Basic": ['trip_days', 'miles', 'receipts'],
    "With penalties": ['trip_days', 'miles', 'receipts', 'has_49', 'has_99'],
    "With ratios": ['trip_days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day'],
    "Polynomial": ['trip_days', 'miles', 'receipts', 'trip_days_squared', 'miles_squared', 'receipts_squared'],
    "Non-linear": ['trip_days', 'miles', 'receipts', 'sqrt_miles', 'sqrt_receipts', 'log_receipts'],
    "Kitchen sink": ['trip_days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day', 
                     'trip_days_squared', 'sqrt_miles', 'log_receipts', 'has_49', 'has_99']
}

best_mae = float('inf')
best_model = None
best_features = None

for name, features in feature_sets.items():
    X = df[features].values
    y = df['expected_output'].values
    
    # Try different regularization
    for alpha in [0, 0.01, 0.1, 1.0]:
        if alpha == 0:
            model = LinearRegression()
        else:
            model = Ridge(alpha=alpha)
        
        model.fit(X, y)
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        
        # Check exact matches
        exact_matches = np.sum(np.abs(predictions - y) < 0.01)
        
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_features = (name, features, alpha)
        
        if exact_matches > 10 or mae < 150:  # Show promising models
            print(f"\n{name} (alpha={alpha}):")
            print(f"  MAE: ${mae:.2f}")
            print(f"  Exact matches: {exact_matches}")
            print(f"  Coefficients:")
            for feat, coef in zip(features, model.coef_):
                print(f"    {feat}: {coef:.6f}")
            print(f"  Intercept: {model.intercept_:.6f}")

# Focus on the best model
print("\n" + "=" * 40)
print("BEST MODEL ANALYSIS")
print("=" * 40)

name, features, alpha = best_features
print(f"\nBest model: {name} (alpha={alpha})")
print(f"Features: {features}")
print(f"MAE: ${best_mae:.2f}")

# Detailed coefficient analysis
X_best = df[features].values
predictions = best_model.predict(X_best)
df['predicted'] = predictions
df['error'] = df['predicted'] - df['expected_output']
df['abs_error'] = np.abs(df['error'])

# Check decimal precision
df['pred_cents'] = (df['predicted'] * 100) % 100
df['exp_cents'] = (df['expected_output'] * 100) % 100
df['cents_error'] = np.abs(df['pred_cents'] - df['exp_cents'])

print(f"\nPrecision analysis:")
print(f"  Exact matches (±$0.01): {(df['abs_error'] < 0.01).sum()}")
print(f"  Close matches (±$1.00): {(df['abs_error'] < 1.00).sum()}")
print(f"  Cents within 5¢: {(df['cents_error'] < 5).sum()}")
print(f"  Cents within 10¢: {(df['cents_error'] < 10).sum()}")

# Try interaction terms
print("\n" + "=" * 40)
print("TESTING INTERACTION TERMS")
print("=" * 40)

# Create polynomial features (degree 2) for basic inputs
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['trip_days', 'miles', 'receipts']].values)
feature_names = poly.get_feature_names_out(['trip_days', 'miles', 'receipts'])

# Fit model with interactions
model_poly = LinearRegression()
model_poly.fit(X_poly, df['expected_output'].values)
pred_poly = model_poly.predict(X_poly)
mae_poly = mean_absolute_error(df['expected_output'].values, pred_poly)
exact_poly = np.sum(np.abs(pred_poly - df['expected_output'].values) < 0.01)

print(f"Polynomial model (degree 2):")
print(f"  MAE: ${mae_poly:.2f}")
print(f"  Exact matches: {exact_poly}")

# Show significant coefficients
significant_coefs = [(name, coef) for name, coef in zip(feature_names, model_poly.coef_) 
                     if abs(coef) > 0.01]
significant_coefs.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\nTop 10 significant terms:")
for name, coef in significant_coefs[:10]:
    print(f"  {name}: {coef:.6f}")

# Analyze specific cases that need precision
print("\n" + "=" * 40)
print("ANALYZING HIGH-PRECISION CASES")
print("=" * 40)

# Look at cases with specific cent patterns
for target_cents in [12, 24, 72, 94]:
    cases = df[df['exp_cents'].round() == target_cents]
    if len(cases) > 0:
        print(f"\nCases ending in .{target_cents} ({len(cases)} total):")
        # Calculate what produces these cents
        case_sample = cases.head(3)
        for idx, row in case_sample.iterrows():
            print(f"  {row['trip_days']:.0f}d, {row['miles']:.0f}mi, ${row['receipts']:.2f} → ${row['expected_output']:.2f}")
            
            # Show contribution of each term
            contribs = []
            for feat, coef in zip(features, best_model.coef_):
                contrib = row[feat] * coef
                if abs(contrib) > 1:
                    contribs.append(f"{feat}*{coef:.4f}={contrib:.2f}")
            print(f"    {' + '.join(contribs[:3])}")

# Save the best model parameters
model_params = {
    'features': features,
    'coefficients': {feat: float(coef) for feat, coef in zip(features, best_model.coef_)},
    'intercept': float(best_model.intercept_),
    'mae': float(best_mae)
}

import json
with open(MODELS_DIR / 'precision_linear_model.json', 'w') as f:
    json.dump(model_params, f, indent=2)

print(f"\nModel parameters saved to: {MODELS_DIR / 'precision_linear_model.json'}") 