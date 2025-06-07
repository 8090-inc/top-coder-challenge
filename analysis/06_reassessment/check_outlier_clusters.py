"""
Check which clusters outliers are being assigned to
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# Load data with cluster assignments
df = pd.read_csv(DATA_DIR / 'public_cases_with_clusters.csv')

# Load predictions to get errors
predictions = pd.read_csv(DATA_DIR / 'predictions' / 'public_cases_predictions_v0.5.csv')
df['abs_error'] = predictions['abs_error']
df['predicted'] = predictions['predicted']

# Find outliers
outliers = df[df['abs_error'] > 500].copy()

print("=" * 80)
print("OUTLIER CLUSTER ANALYSIS")
print("=" * 80)

# Overall cluster distribution
print("\nOverall cluster distribution in dataset:")
for cluster in sorted(df['cluster'].unique()):
    count = (df['cluster'] == cluster).sum()
    pct = count / len(df) * 100
    print(f"Cluster {cluster}: {count} cases ({pct:.1f}%)")

print("\n" + "-" * 40)
print("Outlier distribution by cluster:")
for cluster in sorted(outliers['cluster'].unique()):
    count = (outliers['cluster'] == cluster).sum()
    pct = count / len(outliers) * 100
    print(f"Cluster {cluster}: {count} outliers ({pct:.1f}%)")

# Analyze each problematic pattern
print("\n" + "=" * 80)
print("PATTERN 1: 1-day trips with < 600 miles")
print("=" * 80)

pattern1 = outliers[(outliers['trip_days'] == 1) & (outliers['miles'] < 600)]
print(f"\nTotal: {len(pattern1)} cases")
print("\nCluster assignments:")
print(pattern1['cluster'].value_counts())

print("\nDetailed examples:")
for idx, row in pattern1.head(10).iterrows():
    print(f"\nCluster {row['cluster']}: {row['miles']:.0f} mi, ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected_output']:.2f}, Predicted: ${row['predicted']:.2f}")
    print(f"  Error: ${row['abs_error']:.2f}")

# Check what these SHOULD be
print("\n" + "-" * 40)
print("What outputs these 1-day < 600mi trips SHOULD have:")
all_1day_600 = df[(df['trip_days'] == 1) & (df['miles'] < 600)]
print(f"Mean expected output: ${all_1day_600['expected_output'].mean():.2f}")
print(f"Median expected output: ${all_1day_600['expected_output'].median():.2f}")
print(f"Range: ${all_1day_600['expected_output'].min():.2f} - ${all_1day_600['expected_output'].max():.2f}")

print("\n" + "=" * 80)
print("PATTERN 2: ≤4-day trips with low miles + high receipts")
print("=" * 80)

pattern2 = outliers[(outliers['trip_days'] <= 4) & 
                   (outliers['miles'] < 300) & 
                   (outliers['receipts'] > 1000)]
print(f"\nTotal: {len(pattern2)} cases")

for idx, row in pattern2.iterrows():
    print(f"\nCluster {row['cluster']}: {row['trip_days']:.0f}d, {row['miles']:.0f} mi, ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected_output']:.2f}, Predicted: ${row['predicted']:.2f}")
    print(f"  Error: ${row['abs_error']:.2f}")

# Check Cluster 1 rule issue mentioned in the plan
print("\n" + "=" * 80)
print("CLUSTER 1 ANALYSIS (checking if rule needs tightening)")
print("=" * 80)

cluster1 = df[df['cluster'] == 1]
print(f"\nTotal Cluster 1 cases: {len(cluster1)}")
print(f"Cluster 1 outliers: {(outliers['cluster'] == 1).sum()}")

# Check current rule effectiveness
high_miles_high_receipts = cluster1[(cluster1['miles'] > 600) & (cluster1['receipts'] > 1500)]
print(f"\nCluster 1 cases with miles > 600 AND receipts > $1500: {len(high_miles_high_receipts)}")
print(f"Mean error for these: ${high_miles_high_receipts['abs_error'].mean():.2f}")

# Check alternatives
only_high_miles = cluster1[(cluster1['miles'] > 600) & (cluster1['receipts'] <= 1500)]
print(f"\nCluster 1 cases with miles > 600 BUT receipts <= $1500: {len(only_high_miles)}")
if len(only_high_miles) > 0:
    print(f"Mean error for these: ${only_high_miles['abs_error'].mean():.2f}")

only_high_receipts = cluster1[(cluster1['miles'] <= 600) & (cluster1['receipts'] > 1500)]
print(f"\nCluster 1 cases with miles <= 600 BUT receipts > $1500: {len(only_high_receipts)}")
if len(only_high_receipts) > 0:
    print(f"Mean error for these: ${only_high_receipts['abs_error'].mean():.2f}") 