"""
Deep Analysis of Special Profile Cases
Focus on the 7-8 day, 900-1200 mile, 1000-1200 receipt pattern
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

print("=" * 80)
print("SPECIAL PROFILE DEEP ANALYSIS")
print("=" * 80)

# Load clustered data
df = pd.read_csv(PROCESSED_DATA_DIR / 'public_cases_with_clusters.csv')

# Focus on cluster 5
cluster_5 = df[df['cluster'] == 5]
print(f"\nCluster 5 has {len(cluster_5)} total cases")

# Find special profile cases
special_profile = cluster_5[
    (cluster_5['trip_days'].between(7, 8)) &
    (cluster_5['miles'].between(900, 1200)) &
    (cluster_5['receipts'].between(1000, 1200))
]

print(f"Found {len(special_profile)} special profile cases")

# Detailed analysis of special cases
print("\n" + "=" * 60)
print("SPECIAL PROFILE CASES DETAILED VIEW")
print("=" * 60)

# Sort by output to see pattern
special_sorted = special_profile.sort_values('expected_output')

print("\nAll special profile cases:")
print("-" * 100)
print(f"{'Days':>6} {'Miles':>8} {'Receipts':>10} {'Output':>10} {'Predicted':>10} {'Error':>8}")
print("-" * 100)

for idx, row in special_sorted.iterrows():
    print(f"{row['trip_days']:>6.0f} {row['miles']:>8.0f} {row['receipts']:>10.2f} "
          f"{row['expected_output']:>10.0f} {row['predicted']:>10.0f} {row['residual']:>8.0f}")

# Statistics
print("\nSpecial profile statistics:")
print(f"  Output mean: ${special_profile['expected_output'].mean():.2f}")
print(f"  Output std: ${special_profile['expected_output'].std():.2f}")
print(f"  Output range: ${special_profile['expected_output'].min():.0f} - ${special_profile['expected_output'].max():.0f}")

# Check if output is related to any input
print("\nCorrelations with output:")
for col in ['trip_days', 'miles', 'receipts']:
    if special_profile[col].std() > 0:
        corr = special_profile[col].corr(special_profile['expected_output'])
        print(f"  {col}: {corr:.3f}")

# Try to find a pattern
print("\n" + "=" * 60)
print("SEARCHING FOR PATTERN IN SPECIAL CASES")
print("=" * 60)

# Check if it's a fixed value with noise
outputs = special_profile['expected_output'].values
unique_outputs = np.unique(outputs)
print(f"\nUnique outputs: {len(unique_outputs)}")
if len(unique_outputs) <= 10:
    print("Output values:", unique_outputs)

# Check if it's based on specific receipt values
special_profile['receipt_int'] = special_profile['receipts'].astype(int)
special_profile['receipt_decimal'] = special_profile['receipts'] - special_profile['receipt_int']

print("\nReceipt patterns:")
print(f"  Unique receipt integers: {special_profile['receipt_int'].nunique()}")
print(f"  Unique receipt decimals: {special_profile['receipt_decimal'].nunique()}")

# Group by output to see if there's a pattern
output_groups = special_profile.groupby('expected_output').agg({
    'trip_days': ['count', 'mean'],
    'miles': 'mean',
    'receipts': 'mean'
})
print("\nGrouped by output:")
print(output_groups)

# Compare with non-special cases in cluster 5
print("\n" + "=" * 60)
print("COMPARISON WITH OTHER CLUSTER 5 CASES")
print("=" * 60)

non_special = cluster_5[~cluster_5.index.isin(special_profile.index)]
print(f"\nNon-special cases in cluster 5: {len(non_special)}")

# Find cases that are "almost" special (just outside the boundaries)
almost_special = cluster_5[
    ((cluster_5['trip_days'].between(6, 9)) &
     (cluster_5['miles'].between(800, 1300)) &
     (cluster_5['receipts'].between(900, 1300))) &
    (~cluster_5.index.isin(special_profile.index))
]

print(f"Almost special cases: {len(almost_special)}")

if len(almost_special) > 0:
    print("\nSome 'almost special' cases:")
    print("-" * 100)
    print(f"{'Days':>6} {'Miles':>8} {'Receipts':>10} {'Output':>10} {'Note':>30}")
    print("-" * 100)
    
    for idx, row in almost_special.head(10).iterrows():
        note = []
        if row['trip_days'] < 7 or row['trip_days'] > 8:
            note.append(f"days={row['trip_days']:.0f}")
        if row['miles'] < 900 or row['miles'] > 1200:
            note.append(f"miles={row['miles']:.0f}")
        if row['receipts'] < 1000 or row['receipts'] > 1200:
            note.append(f"receipts=${row['receipts']:.0f}")
        
        print(f"{row['trip_days']:>6.0f} {row['miles']:>8.0f} {row['receipts']:>10.2f} "
              f"{row['expected_output']:>10.0f} {', '.join(note):>30}")

# Visualization
print("\n\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Special vs non-special in 3D space
ax1 = axes[0, 0]
ax1.scatter(non_special['miles'], non_special['receipts'], 
           alpha=0.3, label='Other cluster 5', s=30)
ax1.scatter(special_profile['miles'], special_profile['receipts'], 
           color='red', s=100, label='Special profile', marker='*')
ax1.set_xlabel('Miles')
ax1.set_ylabel('Receipts ($)')
ax1.set_title('Special Profile Cases in Miles-Receipts Space')
ax1.legend()

# Add boundary box
ax1.axvline(x=900, color='red', linestyle='--', alpha=0.5)
ax1.axvline(x=1200, color='red', linestyle='--', alpha=0.5)
ax1.axhline(y=1000, color='red', linestyle='--', alpha=0.5)
ax1.axhline(y=1200, color='red', linestyle='--', alpha=0.5)

# Plot 2: Output distribution
ax2 = axes[0, 1]
ax2.hist(non_special['expected_output'], bins=30, alpha=0.5, label='Other cluster 5')
ax2.hist(special_profile['expected_output'], bins=10, alpha=0.7, label='Special profile', color='red')
ax2.set_xlabel('Expected Output ($)')
ax2.set_ylabel('Count')
ax2.set_title('Output Distribution')
ax2.legend()

# Plot 3: Output vs each input for special cases
ax3 = axes[1, 0]
ax3.scatter(special_profile['receipts'], special_profile['expected_output'], s=100, alpha=0.7)
ax3.set_xlabel('Receipts ($)')
ax3.set_ylabel('Expected Output ($)')
ax3.set_title('Special Profile: Receipts vs Output')

# Add trend line
if len(special_profile) > 1:
    z = np.polyfit(special_profile['receipts'], special_profile['expected_output'], 1)
    p = np.poly1d(z)
    ax3.plot(special_profile['receipts'].sort_values(), 
             p(special_profile['receipts'].sort_values()), 
             "r--", alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.0f}')
    ax3.legend()

# Plot 4: Error analysis
ax4 = axes[1, 1]
ax4.scatter(special_profile['expected_output'], special_profile['residual'], s=100, alpha=0.7, color='red')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.set_xlabel('Expected Output ($)')
ax4.set_ylabel('Model Error ($)')
ax4.set_title('Special Profile: Current Model Errors')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'special_profile_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to '{FIGURES_DIR / 'special_profile_analysis.png'}'")

# Test different formulas for special cases
print("\n" + "=" * 60)
print("TESTING FORMULAS FOR SPECIAL CASES")
print("=" * 60)

if len(special_profile) > 0:
    X = special_profile[['trip_days', 'miles', 'receipts']].values
    y = special_profile['expected_output'].values
    
    # Test 1: Fixed value
    fixed_value = np.median(y)
    fixed_pred = np.full_like(y, fixed_value)
    fixed_mae = mean_absolute_error(y, fixed_pred)
    print(f"\n1. Fixed value ({fixed_value:.0f}): MAE=${fixed_mae:.2f}")
    
    # Test 2: Linear based on receipts only
    if special_profile['receipts'].std() > 0:
        lr_receipts = LinearRegression()
        lr_receipts.fit(special_profile[['receipts']], y)
        receipts_pred = lr_receipts.predict(special_profile[['receipts']])
        receipts_mae = mean_absolute_error(y, receipts_pred)
        print(f"2. Linear (receipts only): MAE=${receipts_mae:.2f}")
        print(f"   Formula: output = {lr_receipts.intercept_:.0f} + {lr_receipts.coef_[0]:.2f}*receipts")
    
    # Test 3: Step function based on receipt ranges
    receipt_bins = [1000, 1050, 1100, 1150, 1200]
    special_profile['receipt_bin'] = pd.cut(special_profile['receipts'], bins=receipt_bins)
    bin_means = special_profile.groupby('receipt_bin')['expected_output'].mean()
    
    step_pred = special_profile['receipt_bin'].map(bin_means)
    step_mae = mean_absolute_error(y, step_pred)
    print(f"3. Step function by receipt bins: MAE=${step_mae:.2f}")
    
    # Show the best approach
    print(f"\nBest approach: Fixed value of ${fixed_value:.0f}")

print("\n" + "=" * 80)
print("SPECIAL PROFILE ANALYSIS COMPLETE")
print("=" * 80) 