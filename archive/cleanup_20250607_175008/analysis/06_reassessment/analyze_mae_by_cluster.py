"""
Analyze MAE contribution by cluster
"""

import pandas as pd
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
from models.cluster_router import assign_cluster_v2

# Load predictions
df = pd.read_csv(DATA_DIR / 'predictions' / 'public_cases_predictions_v3.csv')

# Assign clusters
df['cluster'] = df.apply(lambda row: assign_cluster_v2(row['trip_days'], row['miles'], row['receipts']), axis=1)

# Calculate MAE by cluster
print("=" * 80)
print("MAE CONTRIBUTION BY CLUSTER")
print("=" * 80)

total_mae = 0
total_cases = len(df)

for cluster in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster]
    cluster_mae = cluster_data['abs_error'].mean()
    cluster_cases = len(cluster_data)
    cluster_contribution = (cluster_mae * cluster_cases) / total_cases
    
    print(f"\nCluster {cluster}:")
    print(f"  Cases: {cluster_cases} ({cluster_cases/total_cases*100:.1f}%)")
    print(f"  Cluster MAE: ${cluster_mae:.2f}")
    print(f"  Contribution to total MAE: ${cluster_contribution:.2f}")
    
    # Show worst cases in this cluster
    worst_cases = cluster_data.nlargest(3, 'abs_error')
    if len(worst_cases) > 0:
        print(f"  Worst cases:")
        for idx, row in worst_cases.iterrows():
            print(f"    Case {row['case_id']:.0f}: Error ${row['abs_error']:.2f}")
    
    total_mae += cluster_contribution

print(f"\n{'='*40}")
print(f"Total MAE (sum of contributions): ${total_mae:.2f}")
print(f"Actual MAE: ${df['abs_error'].mean():.2f}")

# Find cases with error > 200 by cluster
print("\n" + "=" * 80)
print("HIGH ERROR CASES (> $200) BY CLUSTER")
print("=" * 80)

high_error = df[df['abs_error'] > 200]
for cluster in sorted(high_error['cluster'].unique()):
    cluster_high = high_error[high_error['cluster'] == cluster]
    print(f"\nCluster {cluster}: {len(cluster_high)} cases with error > $200")
    
    # Sample a few
    for idx, row in cluster_high.head(3).iterrows():
        print(f"  {row['trip_days']:.0f}d, {row['miles']:.0f}mi, ${row['receipts']:.2f} → "
              f"Expected: ${row['expected']:.2f}, Got: ${row['predicted']:.2f}, Error: ${row['abs_error']:.2f}") 