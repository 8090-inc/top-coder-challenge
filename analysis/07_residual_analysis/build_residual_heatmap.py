"""
Build residual heat-map bucketed by various features to identify error patterns
Target: MAE ≤ $60, every cluster ≤ $80
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
from models.cluster_router import assign_cluster_v2

# Load predictions
df = pd.read_csv(DATA_DIR / 'predictions' / 'public_cases_predictions_v3.csv')

# Assign clusters
df['cluster'] = df.apply(lambda row: assign_cluster_v2(row['trip_days'], row['miles'], row['receipts']), axis=1)

# Calculate derived features
df['miles_per_day'] = df['miles'] / df['trip_days']
df['receipts_per_day'] = df['receipts'] / df['trip_days']
df['receipt_bin_50'] = (df['receipts'] // 50) * 50  # $50 bins
df['receipt_bin_100'] = (df['receipts'] // 100) * 100  # $100 bins
df['miles_bin_100'] = (df['miles'] // 100) * 100  # 100 mile bins
df['mpd_bin_50'] = (df['miles_per_day'] // 50) * 50  # 50 miles/day bins

# Calculate residuals and errors
df['residual'] = df['predicted'] - df['expected']
df['pct_error'] = (df['abs_error'] / df['expected']) * 100

print("=" * 80)
print("RESIDUAL HEAT-MAP ANALYSIS")
print("=" * 80)
print(f"\nCurrent Overall MAE: ${df['abs_error'].mean():.2f}")
print(f"Target MAE: $60")
print(f"Gap to target: ${df['abs_error'].mean() - 60:.2f}")

# 1. MAE by Cluster
print("\n" + "=" * 40)
print("MAE BY CLUSTER")
print("=" * 40)
cluster_mae = df.groupby('cluster')['abs_error'].agg(['mean', 'count', 'std']).round(2)
cluster_mae = cluster_mae.sort_values('mean', ascending=False)
print(cluster_mae)
print(f"\nClusters above $80 MAE: {(cluster_mae['mean'] > 80).sum()}")

# 2. Error Heat-map by Cluster and Miles per Day
print("\n" + "=" * 40)
print("ERROR PATTERNS BY CLUSTER AND MILES/DAY")
print("=" * 40)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

clusters = sorted(df['cluster'].unique())
for i, cluster in enumerate(clusters[:9]):
    ax = axes[i]
    cluster_data = df[df['cluster'] == cluster]
    
    # Create pivot table for heat-map
    pivot = cluster_data.pivot_table(
        values='abs_error',
        index='mpd_bin_50',
        columns='receipt_bin_100',
        aggfunc='mean'
    )
    
    if not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'MAE'}, ax=ax)
        ax.set_title(f'Cluster {cluster} (n={len(cluster_data)})')
        ax.set_xlabel('Receipt Bin ($100s)')
        ax.set_ylabel('Miles/Day Bin (50s)')
    else:
        ax.text(0.5, 0.5, f'Cluster {cluster}\nNo data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'residual_heatmap_by_cluster.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Detailed analysis for high-error slices
print("\n" + "=" * 40)
print("HIGH ERROR SLICES (MAE > $200)")
print("=" * 40)

# Find slices with high errors
slices = []
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    
    # By miles per day bins
    for mpd_bin in cluster_data['mpd_bin_50'].unique():
        slice_data = cluster_data[cluster_data['mpd_bin_50'] == mpd_bin]
        if len(slice_data) >= 3:  # At least 3 cases
            mae = slice_data['abs_error'].mean()
            if mae > 200:
                slices.append({
                    'cluster': cluster,
                    'slice_type': 'mpd_bin',
                    'slice_value': mpd_bin,
                    'n_cases': len(slice_data),
                    'mae': mae,
                    'mean_residual': slice_data['residual'].mean()
                })
    
    # By receipt bins
    for receipt_bin in cluster_data['receipt_bin_100'].unique():
        slice_data = cluster_data[cluster_data['receipt_bin_100'] == receipt_bin]
        if len(slice_data) >= 3:
            mae = slice_data['abs_error'].mean()
            if mae > 200:
                slices.append({
                    'cluster': cluster,
                    'slice_type': 'receipt_bin',
                    'slice_value': receipt_bin,
                    'n_cases': len(slice_data),
                    'mae': mae,
                    'mean_residual': slice_data['residual'].mean()
                })

# Sort by MAE
high_error_slices = pd.DataFrame(slices).sort_values('mae', ascending=False)
if len(high_error_slices) > 0:
    print(high_error_slices.head(10).to_string(index=False))
else:
    print("No slices found with MAE > $200")

# 4. Receipt ending analysis
print("\n" + "=" * 40)
print("RECEIPT ENDING PATTERNS")
print("=" * 40)

df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100
ending_analysis = df.groupby('receipt_cents')['abs_error'].agg(['mean', 'count']).round(2)
ending_analysis = ending_analysis[ending_analysis['count'] >= 5]  # At least 5 cases
ending_analysis = ending_analysis.sort_values('mean', ascending=False).head(10)
print(ending_analysis)

# 5. Residual distribution by cluster
print("\n" + "=" * 40)
print("RESIDUAL DISTRIBUTION BY CLUSTER")
print("=" * 40)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, cluster in enumerate(clusters[:9]):
    ax = axes[i]
    cluster_data = df[df['cluster'] == cluster]
    
    if len(cluster_data) > 0:
        ax.hist(cluster_data['residual'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label='Zero error')
        ax.set_title(f'Cluster {cluster} (n={len(cluster_data)})')
        ax.set_xlabel('Residual (Predicted - Expected)')
        ax.set_ylabel('Count')
        
        # Add statistics
        mean_res = cluster_data['residual'].mean()
        std_res = cluster_data['residual'].std()
        ax.text(0.02, 0.98, f'Mean: ${mean_res:.1f}\nStd: ${std_res:.1f}', 
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'residual_distribution_by_cluster.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Systematic bias analysis
print("\n" + "=" * 40)
print("SYSTEMATIC BIAS ANALYSIS")
print("=" * 40)

# Check for systematic over/under prediction
for cluster in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster]
    if len(cluster_data) >= 10:
        mean_residual = cluster_data['residual'].mean()
        pct_over = (cluster_data['residual'] > 0).sum() / len(cluster_data) * 100
        
        if abs(mean_residual) > 50:
            print(f"\nCluster {cluster}:")
            print(f"  Mean residual: ${mean_residual:.2f}")
            print(f"  Over-predicted: {pct_over:.1f}%")
            print(f"  Cases: {len(cluster_data)}")

# 7. Feature importance for high-error cases
print("\n" + "=" * 40)
print("CHARACTERISTICS OF HIGH ERROR CASES")
print("=" * 40)

high_error = df[df['abs_error'] > 200]
low_error = df[df['abs_error'] < 50]

print(f"\nHigh error cases (>{200}): {len(high_error)}")
print(f"Low error cases (<$50): {len(low_error)}")

print("\nMean characteristics:")
features = ['trip_days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day']
comparison = pd.DataFrame({
    'High Error': high_error[features].mean(),
    'Low Error': low_error[features].mean(),
    'Difference': high_error[features].mean() - low_error[features].mean()
}).round(2)
print(comparison)

# Save detailed results
df.to_csv(REPORTS_DIR / 'residual_analysis_detailed.csv', index=False)
print(f"\nDetailed results saved to: {REPORTS_DIR / 'residual_analysis_detailed.csv'}")

# Summary recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR ACHIEVING MAE ≤ $60")
print("=" * 80)

print("\n1. Priority clusters to improve:")
for idx, row in cluster_mae.iterrows():
    if row['mean'] > 80:
        print(f"   - Cluster {idx}: Current MAE ${row['mean']:.2f} (n={row['count']:.0f})")

print("\n2. High-error patterns to address:")
if len(high_error_slices) > 0:
    for _, row in high_error_slices.head(5).iterrows():
        print(f"   - Cluster {row['cluster']}, {row['slice_type']}={row['slice_value']}: MAE ${row['mae']:.2f}")

print("\n3. Systematic biases to correct:")
bias_count = 0
for cluster in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster]
    if len(cluster_data) >= 10:
        mean_residual = cluster_data['residual'].mean()
        if abs(mean_residual) > 50:
            bias_count += 1
            direction = "over" if mean_residual > 0 else "under"
            print(f"   - Cluster {cluster}: {direction}-predicting by ${abs(mean_residual):.2f} on average")

if bias_count == 0:
    print("   - No major systematic biases detected")

print("\n4. Next steps:")
print("   - Implement non-linear transformations for high-error clusters")
print("   - Add interaction terms (miles × receipts, days × receipts)")
print("   - Consider ensemble approach with cluster-specific models")
print("   - Fine-tune receipt ending penalties based on cluster") 