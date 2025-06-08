"""Test V6 Simplified Model"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from model_framework.models.v6_simplified import V6_SimplifiedModel
from models.v5_practical_ensemble import calculate_reimbursement_v5

# Load data
print("Loading data...")
df = pd.read_csv('public_cases_expected_outputs.csv')

# Train model
print("\nTraining V6 Simplified Model...")
model = V6_SimplifiedModel()
model.train(df)

# Evaluate
print("\n=== EVALUATION ===")

v5_errors = []
v6_errors = []

for _, row in df.iterrows():
    actual = row['expected_output']
    
    # V5 prediction
    v5_pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
    v5_error = abs(v5_pred - actual)
    v5_errors.append(v5_error)
    
    # V6 prediction
    v6_pred = model.predict(row['trip_days'], row['miles'], row['receipts'])
    v6_error = abs(v6_pred - actual)
    v6_errors.append(v6_error)

# Calculate metrics
v5_mae = np.mean(v5_errors)
v6_mae = np.mean(v6_errors)

print(f"\nV5 MAE: ${v5_mae:.2f}")
print(f"V6 MAE: ${v6_mae:.2f}")
print(f"Improvement: ${v5_mae - v6_mae:.2f} ({(v5_mae - v6_mae) / v5_mae * 100:.1f}%)")

# Show some examples
print("\n=== SAMPLE PREDICTIONS ===")
for i in range(10):
    row = df.iloc[i]
    v5_pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
    v6_pred = model.predict(row['trip_days'], row['miles'], row['receipts'])
    actual = row['expected_output']
    
    print(f"\nCase {i}: Days={row['trip_days']}, Miles={row['miles']:.0f}, Receipts=${row['receipts']:.2f}")
    print(f"  Actual: ${actual:.2f}")
    print(f"  V5: ${v5_pred:.2f} (error ${abs(v5_pred - actual):.2f})")
    print(f"  V6: ${v6_pred:.2f} (error ${abs(v6_pred - actual):.2f})")
    
    if abs(v6_pred - actual) < abs(v5_pred - actual):
        print("  ✓ V6 better")
    elif abs(v6_pred - actual) > abs(v5_pred - actual):
        print("  ✗ V5 better")
    else:
        print("  = Same") 