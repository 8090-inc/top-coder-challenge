"""
Deep Dive Analysis of Each Cluster
Finding unique calculation patterns for each of the 6 paths
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
sys.path.append(str(Path(__file__).parent.parent.parent))
import calculate_reimbursement as calc

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CLUSTER DEEP DIVE - FINDING UNIQUE CALCULATION PATTERNS")
print("=" * 80)

# Load clustered data
df = pd.read_csv(PROCESSED_DATA_DIR / 'public_cases_with_clusters.csv')
print(f"\nLoaded {len(df)} cases with cluster assignments")

# Add missing column if not present
if 'output_per_day' not in df.columns:
    df['output_per_day'] = df['expected_output'] / df['trip_days']

# Define cluster names based on initial analysis
cluster_names = {
    0: "Standard Multi-Day (5-12 days)",
    1: "Single Day High Miles",
    2: "Long Trip High Receipts",
    3: "Short Trip (3-5 days)",
    4: "Outlier (Low Receipt)",
    5: "Medium Trip High Miles"
}

# Analyze each cluster
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    
    print(f"\n{'=' * 60}")
    print(f"CLUSTER {cluster_id}: {cluster_names[cluster_id]}")
    print(f"Size: {len(cluster_data)} cases")
    print(f"{'=' * 60}")
    
    # 1. Basic statistics
    print("\n1. BASIC STATISTICS:")
    print(f"   Trip days: {cluster_data['trip_days'].mean():.1f} ± {cluster_data['trip_days'].std():.1f}")
    print(f"   Miles: {cluster_data['miles'].mean():.0f} ± {cluster_data['miles'].std():.0f}")
    print(f"   Receipts: ${cluster_data['receipts'].mean():.0f} ± ${cluster_data['receipts'].std():.0f}")
    print(f"   Output: ${cluster_data['expected_output'].mean():.0f} ± ${cluster_data['expected_output'].std():.0f}")
    
    # 2. Ratios and relationships
    print("\n2. KEY RATIOS:")
    print(f"   Miles per day: {cluster_data['miles_per_day'].mean():.1f}")
    print(f"   Receipts per day: ${cluster_data['receipts_per_day'].mean():.0f}")
    print(f"   Output per day: ${cluster_data['output_per_day'].mean():.0f}")
    print(f"   Receipt coverage: {cluster_data['receipt_coverage'].mean():.2f}x")
    
    # 3. Current model performance
    print("\n3. CURRENT MODEL PERFORMANCE:")
    print(f"   MAE: ${cluster_data['residual'].abs().mean():.2f}")
    print(f"   Mean residual: ${cluster_data['residual'].mean():.2f}")
    print(f"   Std residual: ${cluster_data['residual'].std():.2f}")
    
    # Skip cluster 4 (only 1 case)
    if cluster_id == 4:
        print("\n   [Skipping detailed analysis - only 1 case]")
        continue
    
    # 4. Try to find the calculation pattern
    print("\n4. SEARCHING FOR CALCULATION PATTERN:")
    
    # Test different formulas
    X = cluster_data[['trip_days', 'miles', 'receipts']].values
    y = cluster_data['expected_output'].values
    
    # Linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    lr_pred = lr.predict(X)
    lr_mae = mean_absolute_error(y, lr_pred)
    
    print(f"\n   Linear formula: output = {lr.intercept_:.2f} + "
          f"{lr.coef_[0]:.2f}*days + {lr.coef_[1]:.2f}*miles + {lr.coef_[2]:.2f}*receipts")
    print(f"   Linear MAE: ${lr_mae:.2f}")
    
    # Check for special patterns
    # Pattern 1: Fixed daily rate + mileage
    daily_rates = cluster_data['output_per_day'].values
    common_daily_rate = np.median(daily_rates)
    
    # Pattern 2: Receipt-based with cap
    receipt_ratios = cluster_data['receipt_coverage'].values
    common_ratio = np.median(receipt_ratios)
    
    # Pattern 3: Mileage-based
    if cluster_data['miles'].std() > 0:
        miles_coef = np.corrcoef(cluster_data['miles'], cluster_data['expected_output'])[0, 1]
        print(f"\n   Miles correlation: {miles_coef:.3f}")
    
    # Pattern 4: Check for step functions
    if len(cluster_data['trip_days'].unique()) > 1:
        # Group by trip days
        by_days = cluster_data.groupby('trip_days').agg({
            'expected_output': ['mean', 'std', 'count'],
            'miles': 'mean',
            'receipts': 'mean'
        })
        
        print("\n   Output by trip days:")
        for days in sorted(cluster_data['trip_days'].unique())[:5]:  # Show first 5
            day_data = cluster_data[cluster_data['trip_days'] == days]
            if len(day_data) >= 3:  # Only show if enough samples
                print(f"     {days} days: ${day_data['expected_output'].mean():.0f} "
                      f"(n={len(day_data)}, std=${day_data['expected_output'].std():.0f})")
    
    # 5. Look for special cases
    print("\n5. SPECIAL PATTERNS:")
    
    # Check for our known special profile
    special_profile = cluster_data[
        (cluster_data['trip_days'].between(7, 8)) &
        (cluster_data['miles'].between(900, 1200)) &
        (cluster_data['receipts'].between(1000, 1200))
    ]
    if len(special_profile) > 0:
        print(f"   *** CONTAINS {len(special_profile)} SPECIAL PROFILE CASES ***")
        print(f"   Special profile output: ${special_profile['expected_output'].mean():.0f}")
    
    # Check for receipt endings
    receipt_49 = (cluster_data['receipts'] % 1 == 0.49).sum()
    receipt_99 = (cluster_data['receipts'] % 1 == 0.99).sum()
    if receipt_49 > 0 or receipt_99 > 0:
        print(f"   Receipt .49 endings: {receipt_49}")
        print(f"   Receipt .99 endings: {receipt_99}")
    
    # Check for round outputs
    round_outputs = (cluster_data['expected_output'] % 100 == 0).sum()
    if round_outputs > len(cluster_data) * 0.1:
        print(f"   Round outputs ($X00): {round_outputs} ({round_outputs/len(cluster_data)*100:.1f}%)")

# 6. Create visualization comparing clusters
print("\n\nCreating cluster comparison visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot 1: Trip days distribution by cluster
ax1 = axes[0]
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 1:
        ax1.hist(cluster_data['trip_days'], alpha=0.5, label=f'Cluster {cluster_id}', bins=20)
ax1.set_xlabel('Trip Days')
ax1.set_ylabel('Count')
ax1.set_title('Trip Days Distribution by Cluster')
ax1.legend()

# Plot 2: Miles vs Output by cluster
ax2 = axes[1]
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 1:
        ax2.scatter(cluster_data['miles'], cluster_data['expected_output'], 
                   alpha=0.6, label=f'Cluster {cluster_id}', s=30)
ax2.set_xlabel('Miles')
ax2.set_ylabel('Expected Output ($)')
ax2.set_title('Miles vs Output by Cluster')
ax2.legend()

# Plot 3: Receipts vs Output by cluster
ax3 = axes[2]
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 1:
        ax3.scatter(cluster_data['receipts'], cluster_data['expected_output'], 
                   alpha=0.6, label=f'Cluster {cluster_id}', s=30)
ax3.set_xlabel('Receipts ($)')
ax3.set_ylabel('Expected Output ($)')
ax3.set_title('Receipts vs Output by Cluster')
ax3.legend()

# Plot 4: Receipt coverage by cluster
ax4 = axes[3]
receipt_coverage_by_cluster = []
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 1:
        receipt_coverage_by_cluster.append(cluster_data['receipt_coverage'].values)
    else:
        receipt_coverage_by_cluster.append([])

# Filter out empty arrays and cap values for visualization
receipt_coverage_capped = []
labels = []
for i, coverage in enumerate(receipt_coverage_by_cluster):
    if len(coverage) > 0:
        receipt_coverage_capped.append(np.clip(coverage, 0, 5))
        labels.append(f'C{i}')

ax4.boxplot(receipt_coverage_capped, labels=labels)
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Receipt Coverage (capped at 5x)')
ax4.set_title('Receipt Coverage Distribution by Cluster')

# Plot 5: Model error by cluster
ax5 = axes[4]
errors_by_cluster = []
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 1:
        errors_by_cluster.append(abs(cluster_data['residual'].values))

ax5.boxplot(errors_by_cluster, labels=[f'C{i}' for i in range(6) if i != 4])
ax5.set_xlabel('Cluster')
ax5.set_ylabel('Absolute Error ($)')
ax5.set_title('Model Error Distribution by Cluster')

# Plot 6: Cluster characteristics summary
ax6 = axes[5]
ax6.axis('off')
summary_text = "CLUSTER SUMMARY:\n\n"
for cluster_id, name in cluster_names.items():
    cluster_data = df[df['cluster'] == cluster_id]
    summary_text += f"Cluster {cluster_id}: {name}\n"
    summary_text += f"  Size: {len(cluster_data)} cases\n"
    summary_text += f"  MAE: ${abs(cluster_data['residual']).mean():.0f}\n\n"

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cluster_deep_dive.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to '{FIGURES_DIR / 'cluster_deep_dive.png'}'")

# 7. Export detailed cluster analysis
print("\n\nExporting detailed cluster profiles...")

cluster_profiles = []
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    
    profile = {
        'cluster_id': cluster_id,
        'name': cluster_names[cluster_id],
        'size': len(cluster_data),
        'trip_days_mean': cluster_data['trip_days'].mean(),
        'trip_days_mode': cluster_data['trip_days'].mode().values[0] if len(cluster_data['trip_days'].mode()) > 0 else None,
        'miles_mean': cluster_data['miles'].mean(),
        'receipts_mean': cluster_data['receipts'].mean(),
        'output_mean': cluster_data['expected_output'].mean(),
        'mae': abs(cluster_data['residual']).mean(),
        'contains_special_profile': len(cluster_data[
            (cluster_data['trip_days'].between(7, 8)) &
            (cluster_data['miles'].between(900, 1200)) &
            (cluster_data['receipts'].between(1000, 1200))
        ]) > 0
    }
    cluster_profiles.append(profile)

profiles_df = pd.DataFrame(cluster_profiles)
profiles_df.to_csv(REPORTS_DIR / 'cluster_profiles.csv', index=False)
print(f"Saved cluster profiles to '{REPORTS_DIR / 'cluster_profiles.csv'}'")

print("\n" + "=" * 80)
print("CLUSTER DEEP DIVE COMPLETE")
print("=" * 80) 