"""
Identify and analyze outliers with errors > $500
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

print("=" * 80)
print("OUTLIER ANALYSIS - ERRORS > $500")
print("=" * 80)

# Load predictions
df = pd.read_csv(DATA_DIR / 'predictions' / 'public_cases_predictions_v0.5.csv')

# Find outliers
outliers = df[df['abs_error'] > 500].copy()
outliers = outliers.sort_values('abs_error', ascending=False)

print(f"\nFound {len(outliers)} cases with error > $500")
print(f"These represent {len(outliers)/len(df)*100:.1f}% of all cases")

# Analyze patterns
print("\n" + "=" * 40)
print("OUTLIER PATTERNS")
print("=" * 40)

# Group by characteristics
print("\nBy trip days:")
print(outliers['trip_days'].value_counts().head())

print("\nBy error direction:")
print(f"Over-predicted (negative error): {(outliers['error'] < 0).sum()}")
print(f"Under-predicted (positive error): {(outliers['error'] > 0).sum()}")

# Check specific patterns mentioned in the plan
print("\n" + "=" * 40)
print("CHECKING SPECIFIC PATTERNS")
print("=" * 40)

# 1-day trips with < 600 miles
pattern1 = outliers[(outliers['trip_days'] == 1) & (outliers['miles'] < 600)]
print(f"\n1-day trips with < 600 miles: {len(pattern1)} cases")
if len(pattern1) > 0:
    print("Examples:")
    for idx, row in pattern1.head(5).iterrows():
        print(f"  {row['miles']:.0f} mi, ${row['receipts']:.2f} → "
              f"Expected: ${row['expected_output']:.0f}, Got: ${row['predicted']:.0f}, "
              f"Error: ${row['abs_error']:.0f}")

# ≤4-day trips with low miles + high receipts
pattern2 = outliers[(outliers['trip_days'] <= 4) & 
                   (outliers['miles'] < 300) & 
                   (outliers['receipts'] > 1000)]
print(f"\n≤4-day trips with low miles (<300) + high receipts (>$1000): {len(pattern2)} cases")
if len(pattern2) > 0:
    print("Examples:")
    for idx, row in pattern2.head(5).iterrows():
        print(f"  {row['trip_days']:.0f}d, {row['miles']:.0f} mi, ${row['receipts']:.2f} → "
              f"Expected: ${row['expected_output']:.0f}, Got: ${row['predicted']:.0f}, "
              f"Error: ${row['abs_error']:.0f}")

# Analyze all outliers
print("\n" + "=" * 40)
print("ALL OUTLIERS (TOP 20)")
print("=" * 40)

print(f"\n{'Days':>4} {'Miles':>6} {'Receipts':>10} {'Expected':>10} {'Predicted':>10} {'Error':>8}")
print("-" * 60)
for idx, row in outliers.head(20).iterrows():
    print(f"{row['trip_days']:>4.0f} {row['miles']:>6.0f} {row['receipts']:>10.2f} "
          f"{row['expected_output']:>10.2f} {row['predicted']:>10.2f} {row['abs_error']:>8.2f}")

# Check which clusters these fall into
if 'cluster' in df.columns:
    print("\n" + "=" * 40)
    print("OUTLIERS BY CLUSTER")
    print("=" * 40)
    
    outlier_clusters = outliers['cluster'].value_counts()
    for cluster, count in outlier_clusters.items():
        pct = count / len(outliers) * 100
        print(f"Cluster {cluster}: {count} outliers ({pct:.1f}%)")

# Save outlier analysis
outliers.to_csv(REPORTS_DIR / 'outliers_analysis.csv', index=False)
print(f"\nDetailed outlier analysis saved to: {REPORTS_DIR / 'outliers_analysis.csv'}") 