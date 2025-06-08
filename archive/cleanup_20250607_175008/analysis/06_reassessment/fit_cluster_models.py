"""
Fit optimized models for each cluster using the training data
"""

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
from models.cluster_router import assign_cluster_v2

# Load public cases
with open(DATA_DIR / 'raw' / 'public_cases.json', 'r') as f:
    data = json.load(f)

# Create DataFrame with cluster assignments
cases = []
for i, case in enumerate(data):
    trip_days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    cluster = assign_cluster_v2(trip_days, miles, receipts)
    
    cases.append({
        'case_id': i,
        'trip_days': trip_days,
        'miles': miles,
        'receipts': receipts,
        'expected': expected,
        'cluster': cluster,
        'receipt_ends_49': int(receipts * 100) % 100 == 49,
        'receipt_ends_99': int(receipts * 100) % 100 == 99
    })

df = pd.DataFrame(cases)

print("=" * 80)
print("FITTING OPTIMIZED MODELS FOR EACH CLUSTER")
print("=" * 80)

# For each cluster, fit a model
cluster_models = {}

for cluster in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster].copy()
    print(f"\n{'-'*40}")
    print(f"Cluster {cluster}: {len(cluster_data)} cases")
    
    if len(cluster_data) < 5:
        print("Too few cases for regression, using simple average")
        avg_output = cluster_data['expected'].mean()
        print(f"Average output: ${avg_output:.2f}")
        continue
    
    # Features
    X = cluster_data[['trip_days', 'miles', 'receipts']].values
    y = cluster_data['expected'].values
    
    # First, check if receipts ending matters for this cluster
    if cluster_data['receipt_ends_49'].any():
        cases_49 = cluster_data[cluster_data['receipt_ends_49']]
        if len(cases_49) > 0:
            ratio_49 = cases_49['expected'].mean() / cluster_data['expected'].mean()
            print(f"Receipt .49 penalty ratio: {ratio_49:.3f}")
    
    if cluster_data['receipt_ends_99'].any():
        cases_99 = cluster_data[cluster_data['receipt_ends_99']]
        if len(cases_99) > 0:
            ratio_99 = cases_99['expected'].mean() / cluster_data['expected'].mean()
            print(f"Receipt .99 penalty ratio: {ratio_99:.3f}")
    
    # Try different models
    # 1. Linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    lr_pred = lr.predict(X)
    lr_mae = mean_absolute_error(y, lr_pred)
    
    print(f"\nLinear Regression:")
    print(f"  Intercept: {lr.intercept_:.2f}")
    print(f"  Coefficients: days={lr.coef_[0]:.2f}, miles={lr.coef_[1]:.3f}, receipts={lr.coef_[2]:.3f}")
    print(f"  MAE: ${lr_mae:.2f}")
    
    # 2. Ridge regression (helps with multicollinearity)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    ridge_pred = ridge.predict(X)
    ridge_mae = mean_absolute_error(y, ridge_pred)
    
    print(f"\nRidge Regression:")
    print(f"  MAE: ${ridge_mae:.2f}")
    
    # 3. Simple decision tree
    if len(cluster_data) > 20:
        dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)
        dt.fit(X, y)
        dt_pred = dt.predict(X)
        dt_mae = mean_absolute_error(y, dt_pred)
        
        print(f"\nDecision Tree:")
        print(f"  MAE: ${dt_mae:.2f}")
        print(f"  Tree depth: {dt.get_depth()}")
    
    # Store the best model
    cluster_models[cluster] = {
        'model': lr,
        'type': 'linear',
        'mae': lr_mae,
        'n_cases': len(cluster_data)
    }

# Generate Python code for the best models
print("\n" + "=" * 80)
print("GENERATING OPTIMIZED PYTHON FUNCTIONS")
print("=" * 80)

for cluster, model_info in cluster_models.items():
    print(f"\ndef calculate_cluster_{cluster}_optimized(trip_days, miles, receipts):")
    
    if model_info['type'] == 'linear':
        lr = model_info['model']
        print(f"    return {lr.intercept_:.2f} + {lr.coef_[0]:.2f} * trip_days + " +
              f"{lr.coef_[1]:.3f} * miles + {lr.coef_[2]:.3f} * receipts")
    
    print() 