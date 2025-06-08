import pandas as pd
import numpy as np
import json

# Load predictions and actual values
df = pd.read_csv('../public_cases_predictions_v5.csv')
df['error'] = np.abs(df['predicted'] - df['expected_output'])
df['error_pct'] = df['error'] / df['expected_output'] * 100

# Sort by absolute error
df_sorted = df.sort_values('error', ascending=False)

print("V5 MODEL ERROR ANALYSIS")
print("=" * 60)
print(f"\nOverall Statistics:")
print(f"MAE: ${df['error'].mean():.2f}")
print(f"MAPE: {df['error_pct'].mean():.1f}%")
print(f"Max Error: ${df['error'].max():.2f}")
print(f"Cases with error > $200: {(df['error'] > 200).sum()}")
print(f"Cases with error > $100: {(df['error'] > 100).sum()}")

print("\n\nTop 20 Largest Errors:")
print("-" * 120)
print(f"{'Days':>5} {'Miles':>7} {'Receipts':>10} {'Expected':>10} {'Predicted':>10} {'Error':>8} {'Error%':>7} {'Cluster':>8}")
print("-" * 120)

for idx, row in df_sorted.head(20).iterrows():
    print(f"{row['trip_days']:5.0f} {row['miles']:7.0f} ${row['receipts']:9.2f} "
          f"${row['expected_output']:9.2f} ${row['predicted']:9.2f} "
          f"${row['error']:7.2f} {row['error_pct']:6.1f}% {row.get('cluster', 'N/A'):>8}")

# Analyze patterns in high-error cases
high_error = df[df['error'] > 100]

print(f"\n\nHigh Error Cases Analysis (>{100}):")
print(f"Total cases: {len(high_error)}")

# Check for receipt ending patterns
high_error['receipt_cents'] = (high_error['receipts'] * 100).astype(int) % 100
print(f"\nReceipt endings in high-error cases:")
print(high_error['receipt_cents'].value_counts().head(10))

# Check for cluster distribution
if 'cluster' in high_error.columns:
    print(f"\nCluster distribution in high-error cases:")
    print(high_error['cluster'].value_counts().sort_index())

# Look for specific patterns
print("\n\nPattern Analysis:")

# Very high receipts
very_high_receipts = high_error[high_error['receipts'] > 2000]
if len(very_high_receipts) > 0:
    print(f"\nVery high receipts (>${2000}): {len(very_high_receipts)} cases")
    print(f"Average error: ${very_high_receipts['error'].mean():.2f}")

# Single day trips
single_day = high_error[high_error['trip_days'] == 1]
if len(single_day) > 0:
    print(f"\nSingle day trips: {len(single_day)} cases")
    print(f"Average error: ${single_day['error'].mean():.2f}")

# Long trips
long_trips = high_error[high_error['trip_days'] >= 10]
if len(long_trips) > 0:
    print(f"\nLong trips (>=10 days): {len(long_trips)} cases")
    print(f"Average error: ${long_trips['error'].mean():.2f}")

# Export high error cases for manual review
high_error_export = df_sorted.head(50)[['trip_days', 'miles', 'receipts', 
                                        'expected_output', 'predicted', 'error', 'error_pct']]
high_error_export.to_csv('../high_error_cases_v5.csv', index=False)
print(f"\n\nExported top 50 error cases to 'high_error_cases_v5.csv' for manual review") 