import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from models.cluster_models_optimized import (
    calculate_cluster_0_optimized, calculate_cluster_1a_optimized,
    calculate_cluster_1b_optimized, calculate_cluster_2_optimized,
    calculate_cluster_3_optimized, calculate_cluster_4_optimized,
    calculate_cluster_5_optimized, calculate_cluster_6_optimized
)
from models.cluster_router import assign_cluster_v2

# Load data
df = pd.read_csv('../public_cases_expected_outputs.csv')
# Rename columns to match expected names
df = df.rename(columns={
    'trip_days': 'trip_days',
    'miles': 'miles_traveled', 
    'receipts': 'total_receipts_amount',
    'expected_output': 'expected_reimbursement'
})
df['receipt_cents'] = (df['total_receipts_amount'] * 100).astype(int) % 100

print("CALIBRATING RECEIPT ENDING PENALTIES")
print("=" * 60)

# Function to get base amount without penalty
def get_base_amount(row):
    cluster = assign_cluster_v2(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    
    if cluster == '0':
        return calculate_cluster_0_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    elif cluster == '1a':
        return calculate_cluster_1a_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    elif cluster == '1b':
        return calculate_cluster_1b_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    elif cluster == '2':
        return calculate_cluster_2_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    elif cluster == '3':
        return calculate_cluster_3_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    elif cluster == '4':
        return calculate_cluster_4_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    elif cluster == '5':
        return calculate_cluster_5_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    elif cluster == '6':
        return calculate_cluster_6_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])
    else:
        return calculate_cluster_0_optimized(row['trip_days'], row['miles_traveled'], row['total_receipts_amount'])

# Analyze .99 endings
ending_99 = df[df['receipt_cents'] == 99].copy()
print(f"\nAnalyzing {len(ending_99)} cases with .99 endings...")

# Calculate base amounts and actual penalties
ending_99['base_amount'] = ending_99.apply(get_base_amount, axis=1)
ending_99['actual_penalty_factor'] = ending_99['expected_reimbursement'] / ending_99['base_amount']

print("\n.99 Ending Analysis:")
print(f"  Current penalty factor: 0.51")
print(f"  Mean actual penalty factor: {ending_99['actual_penalty_factor'].mean():.3f}")
print(f"  Median actual penalty factor: {ending_99['actual_penalty_factor'].median():.3f}")
print(f"  Std dev: {ending_99['actual_penalty_factor'].std():.3f}")

# Show distribution
print("\n  Distribution of actual penalty factors:")
for percentile in [10, 25, 50, 75, 90]:
    val = ending_99['actual_penalty_factor'].quantile(percentile/100)
    print(f"    {percentile}th percentile: {val:.3f}")

# Analyze .49 endings
ending_49 = df[df['receipt_cents'] == 49].copy()
print(f"\n\nAnalyzing {len(ending_49)} cases with .49 endings...")

ending_49['base_amount'] = ending_49.apply(get_base_amount, axis=1)
ending_49['actual_penalty_factor'] = ending_49['expected_reimbursement'] / ending_49['base_amount']

print("\n.49 Ending Analysis:")
print(f"  Current penalty factor: 0.341")
print(f"  Mean actual penalty factor: {ending_49['actual_penalty_factor'].mean():.3f}")
print(f"  Median actual penalty factor: {ending_49['actual_penalty_factor'].median():.3f}")
print(f"  Std dev: {ending_49['actual_penalty_factor'].std():.3f}")

# Show distribution
print("\n  Distribution of actual penalty factors:")
for percentile in [10, 25, 50, 75, 90]:
    val = ending_49['actual_penalty_factor'].quantile(percentile/100)
    print(f"    {percentile}th percentile: {val:.3f}")

# Check for patterns by cluster
print("\n\nPenalty Factors by Cluster:")
print("-" * 60)

for cents, cases in [('.99', ending_99), ('.49', ending_49)]:
    print(f"\n{cents} endings by cluster:")
    cases['cluster'] = cases.apply(
        lambda r: assign_cluster_v2(r['trip_days'], r['miles_traveled'], r['total_receipts_amount']), 
        axis=1
    )
    
    cluster_stats = cases.groupby('cluster')['actual_penalty_factor'].agg(['mean', 'count', 'std'])
    for cluster, stats in cluster_stats.iterrows():
        if stats['count'] > 0:
            print(f"  Cluster {cluster}: mean={stats['mean']:.3f}, n={int(stats['count'])}, std={stats['std']:.3f}")

# Calculate optimal penalty factors
print("\n\n" + "=" * 60)
print("RECOMMENDED PENALTY FACTORS:")
print("=" * 60)

# For .99 endings - use a more conservative factor
optimal_99 = ending_99['actual_penalty_factor'].median()  # Use median to avoid outliers
print(f"\n.99 endings:")
print(f"  Current: 0.51")
print(f"  Recommended: {optimal_99:.3f}")
print(f"  Expected MAE reduction: ~${(0.51 - optimal_99) * ending_99['base_amount'].mean():.2f} per case")

# For .49 endings
optimal_49 = ending_49['actual_penalty_factor'].median()
print(f"\n.49 endings:")
print(f"  Current: 0.341")  
print(f"  Recommended: {optimal_49:.3f}")
print(f"  Expected MAE reduction: ~${abs(0.341 - optimal_49) * ending_49['base_amount'].mean():.2f} per case")

# Calculate total impact
total_impact = (
    len(ending_99) * (0.51 - optimal_99) * ending_99['base_amount'].mean() +
    len(ending_49) * abs(0.341 - optimal_49) * ending_49['base_amount'].mean()
) / len(df)

print(f"\n\nTotal expected MAE improvement: ~${total_impact:.2f}")
print(f"As percentage of current MAE ($77.41): {total_impact/77.41*100:.1f}%") 