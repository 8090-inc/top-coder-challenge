import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from models.cluster_models_optimized import calculate_reimbursement_v3
from models.cluster_router import assign_cluster_v2

# Load data
df = pd.read_csv('../public_cases_expected_outputs.csv')

# Calculate rule engine predictions
df['rule_pred'] = df.apply(
    lambda r: calculate_reimbursement_v3(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)
df['rule_error'] = df['rule_pred'] - df['expected_output']
df['cluster'] = df.apply(
    lambda r: assign_cluster_v2(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Rule Engine Error Analysis', fontsize=16)

# 1. Error distribution by cluster
ax1 = axes[0, 0]
cluster_errors = df.groupby('cluster')['rule_error'].apply(list)
ax1.boxplot([errors for errors in cluster_errors], labels=cluster_errors.index)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Prediction Error ($)')
ax1.set_title('Error Distribution by Cluster')
ax1.grid(True, alpha=0.3)

# 2. Actual vs Predicted scatter
ax2 = axes[0, 1]
scatter = ax2.scatter(df['expected_output'], df['rule_pred'], 
                     c=df['cluster'].astype('category').cat.codes, 
                     alpha=0.6, cmap='tab10')
ax2.plot([0, 3000], [0, 3000], 'r--', alpha=0.5, label='Perfect prediction')
ax2.set_xlabel('Expected Output ($)')
ax2.set_ylabel('Rule Engine Prediction ($)')
ax2.set_title('Actual vs Predicted')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Error vs Trip Days
ax3 = axes[1, 0]
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    if len(cluster_data) > 10:  # Only plot clusters with enough data
        ax3.scatter(cluster_data['trip_days'], cluster_data['rule_error'], 
                   label=f'Cluster {cluster}', alpha=0.6)
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Trip Days')
ax3.set_ylabel('Prediction Error ($)')
ax3.set_title('Error vs Trip Duration')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. Error heatmap by trip characteristics
ax4 = axes[1, 1]
# Create bins for heatmap
df['days_bin'] = pd.cut(df['trip_days'], bins=[0, 1, 5, 10, 20], labels=['1', '2-5', '6-10', '11+'])
df['miles_bin'] = pd.cut(df['miles'], bins=[0, 200, 500, 1000, 2000], labels=['0-200', '201-500', '501-1000', '1000+'])

# Calculate mean absolute error for each bin combination
heatmap_data = df.groupby(['days_bin', 'miles_bin'])['rule_error'].apply(lambda x: np.abs(x).mean()).unstack()
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'MAE ($)'})
ax4.set_xlabel('Miles Range')
ax4.set_ylabel('Days Range')
ax4.set_title('Mean Absolute Error by Trip Characteristics')

plt.tight_layout()
plt.savefig('rule_engine_errors.png', dpi=150, bbox_inches='tight')
print("Saved visualization to rule_engine_errors.png")

# Print insights
print("\nKEY INSIGHTS FROM VISUALIZATION:")
print("=" * 60)

# Cluster biases
print("\n1. Systematic Cluster Biases:")
cluster_biases = df.groupby('cluster')['rule_error'].mean().sort_values()
for cluster, bias in cluster_biases.items():
    if abs(bias) > 20:
        direction = "underpredicts" if bias < 0 else "overpredicts"
        print(f"   Cluster {cluster} {direction} by ${abs(bias):.0f} on average")

# High error regions
print("\n2. High Error Regions (from heatmap):")
for days in heatmap_data.index:
    for miles in heatmap_data.columns:
        if pd.notna(heatmap_data.loc[days, miles]) and heatmap_data.loc[days, miles] > 150:
            print(f"   {days} days, {miles} miles: MAE ${heatmap_data.loc[days, miles]:.0f}")

# Outlier patterns
print("\n3. Outlier Patterns:")
outliers = df[np.abs(df['rule_error']) > 300]
if len(outliers) > 0:
    print(f"   {len(outliers)} cases with error > $300")
    print(f"   Common characteristics:")
    print(f"     - Average days: {outliers['trip_days'].mean():.1f}")
    print(f"     - Average miles: {outliers['miles'].mean():.0f}")
    print(f"     - Average receipts: ${outliers['receipts'].mean():.0f}")
    print(f"     - Most common cluster: {outliers['cluster'].mode()[0] if len(outliers['cluster'].mode()) > 0 else 'N/A'}")

plt.show() 