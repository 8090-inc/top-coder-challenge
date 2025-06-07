"""
Test the v2 model performance on public cases
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.cluster_models import calculate_reimbursement_v2

# Load public cases
with open(DATA_DIR / 'raw' / 'public_cases.json', 'r') as f:
    data = json.load(f)

# Test all cases
results = []
for i, case in enumerate(data):
    trip_days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement_v2(trip_days, miles, receipts)
    error = predicted - expected
    abs_error = abs(error)
    
    results.append({
        'case_id': i,
        'trip_days': trip_days,
        'miles': miles,
        'receipts': receipts,
        'expected': expected,
        'predicted': predicted,
        'error': error,
        'abs_error': abs_error
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Calculate metrics
mae = df_results['abs_error'].mean()
mape = (df_results['abs_error'] / df_results['expected']).mean() * 100
max_error = df_results['abs_error'].max()
errors_over_500 = (df_results['abs_error'] > 500).sum()
errors_over_110 = (df_results['abs_error'] > 110).sum()

print("=" * 80)
print("V2 MODEL PERFORMANCE EVALUATION")
print("=" * 80)

print(f"\nOverall Metrics:")
print(f"  MAE: ${mae:.2f}")
print(f"  MAPE: {mape:.1f}%")
print(f"  Max Error: ${max_error:.2f}")
print(f"  Cases with error > $500: {errors_over_500} ({errors_over_500/len(df_results)*100:.1f}%)")
print(f"  Cases with error > $110: {errors_over_110} ({errors_over_110/len(df_results)*100:.1f}%)")

# Check against exit criteria
print(f"\n{'='*40}")
print("EXIT CRITERIA CHECK:")
print(f"{'='*40}")
print(f"  Target MAE ≤ $110: {'✅ PASS' if mae <= 110 else '❌ FAIL'} (Current: ${mae:.2f})")
print(f"  No error > $500: {'✅ PASS' if errors_over_500 == 0 else '❌ FAIL'} ({errors_over_500} violations)")

# Analyze remaining outliers
if errors_over_500 > 0:
    print("\n" + "=" * 40)
    print("REMAINING OUTLIERS (Error > $500):")
    print("=" * 40)
    
    outliers = df_results[df_results['abs_error'] > 500].sort_values('abs_error', ascending=False)
    
    from models.cluster_router import assign_cluster_v2
    
    for idx, row in outliers.head(10).iterrows():
        cluster = assign_cluster_v2(row['trip_days'], row['miles'], row['receipts'])
        print(f"\nCase {row['case_id']}: Cluster {cluster}")
        print(f"  {row['trip_days']}d, {row['miles']:.0f}mi, ${row['receipts']:.2f}")
        print(f"  Expected: ${row['expected']:.2f}, Predicted: ${row['predicted']:.2f}")
        print(f"  Error: ${row['abs_error']:.2f}")

# Save results
df_results.to_csv(DATA_DIR / 'predictions' / 'public_cases_predictions_v2.csv', index=False)
print(f"\nPredictions saved to: {DATA_DIR / 'predictions' / 'public_cases_predictions_v2.csv'}")

# Error distribution
print("\n" + "=" * 40)
print("ERROR DISTRIBUTION:")
print("=" * 40)
print(f"  < $50: {(df_results['abs_error'] < 50).sum()} cases ({(df_results['abs_error'] < 50).sum()/len(df_results)*100:.1f}%)")
print(f"  $50-$100: {((df_results['abs_error'] >= 50) & (df_results['abs_error'] < 100)).sum()} cases")
print(f"  $100-$200: {((df_results['abs_error'] >= 100) & (df_results['abs_error'] < 200)).sum()} cases")
print(f"  $200-$500: {((df_results['abs_error'] >= 200) & (df_results['abs_error'] < 500)).sum()} cases")
print(f"  > $500: {(df_results['abs_error'] >= 500).sum()} cases") 