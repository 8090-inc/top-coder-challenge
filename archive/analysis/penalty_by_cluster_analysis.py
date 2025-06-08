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
df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100

# Focus on penalty cases
penalty_cases = df[df['receipt_cents'].isin([49, 99])].copy()

print("ANALYZING PENALTY FACTORS BY CLUSTER")
print("=" * 60)

# Get cluster and base amount for each case
results = []
for _, row in penalty_cases.iterrows():
    cluster = assign_cluster_v2(row['trip_days'], row['miles'], row['receipts'])
    
    # Get base amount
    if cluster == '0':
        base = calculate_cluster_0_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '1a':
        base = calculate_cluster_1a_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '1b':
        base = calculate_cluster_1b_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '2':
        base = calculate_cluster_2_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '3':
        base = calculate_cluster_3_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '4':
        base = calculate_cluster_4_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '5':
        base = calculate_cluster_5_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '6':
        base = calculate_cluster_6_optimized(row['trip_days'], row['miles'], row['receipts'])
    else:
        base = calculate_cluster_0_optimized(row['trip_days'], row['miles'], row['receipts'])
    
    actual_factor = row['expected_output'] / base if base > 0 else 0
    
    results.append({
        'cluster': cluster,
        'cents': int(row['receipt_cents']),
        'base_amount': base,
        'expected': row['expected_output'],
        'actual_factor': actual_factor,
        'trip_days': row['trip_days'],
        'miles': row['miles'],
        'receipts': row['receipts']
    })

results_df = pd.DataFrame(results)

# Analyze by cluster and cents type
print("\nPenalty Factors by Cluster:")
print("-" * 80)

for cents in [49, 99]:
    print(f"\n{cents} cent endings:")
    cent_data = results_df[results_df['cents'] == cents]
    
    cluster_stats = cent_data.groupby('cluster')['actual_factor'].agg(['mean', 'std', 'count', 'min', 'max'])
    
    for cluster, stats in cluster_stats.iterrows():
        print(f"  Cluster {cluster}:")
        print(f"    Mean factor: {stats['mean']:.3f}")
        print(f"    Std dev: {stats['std']:.3f}")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Count: {int(stats['count'])}")
        
        # Show individual cases if few
        if stats['count'] <= 3:
            cluster_cases = cent_data[cent_data['cluster'] == cluster]
            for _, case in cluster_cases.iterrows():
                print(f"      {case['trip_days']:.0f}d, {case['miles']:.0f}mi, ${case['receipts']:.2f} → factor={case['actual_factor']:.3f}")

# Check for outliers
print("\n\nOutlier Analysis:")
print("-" * 60)

# Find cases with very different factors
for cents in [49, 99]:
    cent_data = results_df[results_df['cents'] == cents]
    median_factor = cent_data['actual_factor'].median()
    
    outliers = cent_data[
        (cent_data['actual_factor'] < median_factor * 0.7) | 
        (cent_data['actual_factor'] > median_factor * 1.3)
    ]
    
    if len(outliers) > 0:
        print(f"\n{cents} cent outliers (>30% from median {median_factor:.3f}):")
        for _, case in outliers.iterrows():
            print(f"  Cluster {case['cluster']}: {case['trip_days']:.0f}d, {case['miles']:.0f}mi, ${case['receipts']:.2f}")
            print(f"    Factor: {case['actual_factor']:.3f} (expected ${case['expected']:.2f}, base ${case['base_amount']:.2f})")

# Recommendations
print("\n\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

# Calculate optimal factors per cluster
print("\nCluster-specific penalty factors:")
for cents in [49, 99]:
    print(f"\n{cents} cent endings:")
    cent_data = results_df[results_df['cents'] == cents]
    
    # Global factor
    global_median = cent_data['actual_factor'].median()
    print(f"  Global median: {global_median:.3f}")
    
    # Per-cluster factors
    cluster_stats = cent_data.groupby('cluster')['actual_factor'].agg(['median', 'count'])
    for cluster, stats in cluster_stats.iterrows():
        if stats['count'] >= 2:  # Only if we have enough data
            print(f"  Cluster {cluster}: {stats['median']:.3f} (n={int(stats['count'])})")

print("\n\nConclusion:")
if results_df.groupby(['cluster', 'cents'])['actual_factor'].std().mean() < 0.1:
    print("✓ Penalty factors are consistent within clusters - use cluster-specific factors")
else:
    print("✗ High variance even within clusters - may need more complex logic") 