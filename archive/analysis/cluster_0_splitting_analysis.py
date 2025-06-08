import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../public_cases_expected_outputs.csv')

# Get cluster 0 cases
import sys
sys.path.append('..')
from models.cluster_router import assign_cluster_v2

df['cluster'] = df.apply(lambda r: assign_cluster_v2(r['trip_days'], r['miles'], r['receipts']), axis=1)
cluster_0 = df[df['cluster'] == '0'].copy()

print("CLUSTER 0 SPLITTING ANALYSIS")
print("=" * 60)
print(f"Total Cluster 0 cases: {len(cluster_0)}")

# Add derived features
cluster_0['miles_per_day'] = cluster_0['miles'] / cluster_0['trip_days']
cluster_0['receipts_per_day'] = cluster_0['receipts'] / cluster_0['trip_days']
cluster_0['receipt_cents'] = (cluster_0['receipts'] * 100).astype(int) % 100

# Calculate current errors for comparison
from models.cluster_models_optimized import calculate_cluster_0_optimized
cluster_0['current_pred'] = cluster_0.apply(
    lambda r: calculate_cluster_0_optimized(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)
# Apply penalties
cluster_0['penalty_factor'] = 1.0
cluster_0.loc[cluster_0['receipt_cents'] == 49, 'penalty_factor'] = 0.341
cluster_0.loc[cluster_0['receipt_cents'] == 99, 'penalty_factor'] = 0.51
cluster_0['current_pred_final'] = cluster_0['current_pred'] * cluster_0['penalty_factor']
cluster_0['current_error'] = np.abs(cluster_0['current_pred_final'] - cluster_0['expected_output'])

print(f"Current MAE: ${cluster_0['current_error'].mean():.2f}")

# Try different splitting strategies
print("\n\nEXPLORING SPLITTING STRATEGIES:")
print("-" * 60)

# Strategy 1: Trip duration based
print("\n1. DURATION-BASED SPLITS:")
duration_groups = {
    'Short (2-4 days)': cluster_0[cluster_0['trip_days'].between(2, 4)],
    'Medium (5-8 days)': cluster_0[cluster_0['trip_days'].between(5, 8)],
    'Long (9-14 days)': cluster_0[cluster_0['trip_days'] >= 9]
}

for name, group in duration_groups.items():
    if len(group) > 0:
        print(f"\n{name}: {len(group)} cases")
        print(f"  MAE: ${group['current_error'].mean():.2f}")
        print(f"  Characteristics: {group['miles'].mean():.0f} mi, ${group['receipts'].mean():.0f} receipts")

# Strategy 2: Efficiency based
print("\n\n2. EFFICIENCY-BASED SPLITS:")
efficiency_groups = {
    'Low efficiency (<50 mi/day)': cluster_0[cluster_0['miles_per_day'] < 50],
    'Normal efficiency (50-150 mi/day)': cluster_0[cluster_0['miles_per_day'].between(50, 150)],
    'High efficiency (>150 mi/day)': cluster_0[cluster_0['miles_per_day'] > 150]
}

for name, group in efficiency_groups.items():
    if len(group) > 0:
        print(f"\n{name}: {len(group)} cases")
        print(f"  MAE: ${group['current_error'].mean():.2f}")

# Strategy 3: K-means clustering
print("\n\n3. K-MEANS CLUSTERING:")
features_for_clustering = ['trip_days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day']
X = cluster_0[features_for_clustering]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different numbers of clusters
best_k = 3
best_score = float('inf')

for k in range(3, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_0[f'kmeans_{k}'] = kmeans.fit_predict(X_scaled)
    
    # Calculate MAE for this clustering
    total_mae = 0
    for i in range(k):
        sub_cluster = cluster_0[cluster_0[f'kmeans_{k}'] == i]
        if len(sub_cluster) > 5:  # Need enough samples
            total_mae += sub_cluster['current_error'].mean() * len(sub_cluster)
    
    avg_mae = total_mae / len(cluster_0)
    print(f"\nK={k}: Average MAE ${avg_mae:.2f}")
    
    if avg_mae < best_score:
        best_score = avg_mae
        best_k = k

# Use best k-means clustering
cluster_0['sub_cluster'] = cluster_0[f'kmeans_{best_k}']

# Strategy 4: Hybrid approach (combine insights)
print("\n\n4. HYBRID APPROACH (RECOMMENDED):")
print("-" * 60)

def assign_sub_cluster_hybrid(row):
    """Assign sub-cluster based on multiple criteria"""
    days = row['trip_days']
    miles = row['miles']
    receipts = row['receipts']
    mpd = row['miles_per_day']
    
    # Sub-cluster 0a: Short trips (2-4 days)
    if 2 <= days <= 4:
        return '0a_short'
    
    # Sub-cluster 0b: Low efficiency trips
    elif mpd < 50 and days >= 5:
        return '0b_low_efficiency'
    
    # Sub-cluster 0c: Long trips (9+ days)
    elif days >= 9:
        return '0c_long'
    
    # Sub-cluster 0d: High receipt trips
    elif receipts > 1500 and days >= 5:
        return '0d_high_receipt'
    
    # Sub-cluster 0e: Standard trips (5-8 days, normal patterns)
    else:
        return '0e_standard'

cluster_0['hybrid_sub_cluster'] = cluster_0.apply(assign_sub_cluster_hybrid, axis=1)

# Analyze hybrid approach
print("\nHybrid sub-cluster distribution:")
for sub_cluster in sorted(cluster_0['hybrid_sub_cluster'].unique()):
    sub_data = cluster_0[cluster_0['hybrid_sub_cluster'] == sub_cluster]
    print(f"\n{sub_cluster}: {len(sub_data)} cases ({len(sub_data)/len(cluster_0)*100:.1f}%)")
    print(f"  Current MAE: ${sub_data['current_error'].mean():.2f}")
    print(f"  Avg days: {sub_data['trip_days'].mean():.1f}")
    print(f"  Avg miles: {sub_data['miles'].mean():.0f}")
    print(f"  Avg receipts: ${sub_data['receipts'].mean():.0f}")
    print(f"  Avg mi/day: {sub_data['miles_per_day'].mean():.0f}")

# Fit optimized models for each sub-cluster
print("\n\nOPTIMIZED LINEAR MODELS FOR SUB-CLUSTERS:")
print("-" * 60)

sub_cluster_models = {}
total_new_mae = 0

for sub_cluster in sorted(cluster_0['hybrid_sub_cluster'].unique()):
    sub_data = cluster_0[cluster_0['hybrid_sub_cluster'] == sub_cluster]
    
    if len(sub_data) > 10:  # Need enough data
        # Fit linear model
        X = sub_data[['trip_days', 'miles', 'receipts']]
        y = sub_data['expected_output'] / sub_data['penalty_factor']  # Remove penalty effect
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Store model
        sub_cluster_models[sub_cluster] = {
            'intercept': model.intercept_,
            'coef_days': model.coef_[0],
            'coef_miles': model.coef_[1],
            'coef_receipts': model.coef_[2],
            'n_samples': len(sub_data)
        }
        
        # Calculate new predictions
        sub_data['new_pred'] = model.predict(X)
        sub_data['new_pred_final'] = sub_data['new_pred'] * sub_data['penalty_factor']
        sub_data['new_error'] = np.abs(sub_data['new_pred_final'] - sub_data['expected_output'])
        
        old_mae = sub_data['current_error'].mean()
        new_mae = sub_data['new_error'].mean()
        improvement = old_mae - new_mae
        
        print(f"\n{sub_cluster}:")
        print(f"  Formula: {model.intercept_:.2f} + {model.coef_[0]:.2f}*days + "
              f"{model.coef_[1]:.3f}*miles + {model.coef_[2]:.3f}*receipts")
        print(f"  Old MAE: ${old_mae:.2f}")
        print(f"  New MAE: ${new_mae:.2f}")
        print(f"  Improvement: ${improvement:.2f}")
        
        total_new_mae += new_mae * len(sub_data)
    else:
        # Use original formula for small sub-clusters
        total_new_mae += sub_data['current_error'].mean() * len(sub_data)

overall_new_mae = total_new_mae / len(cluster_0)
print(f"\n\nOVERALL RESULTS:")
print(f"Current Cluster 0 MAE: ${cluster_0['current_error'].mean():.2f}")
print(f"New split model MAE: ${overall_new_mae:.2f}")
print(f"Improvement: ${cluster_0['current_error'].mean() - overall_new_mae:.2f}")
print(f"Improvement %: {(cluster_0['current_error'].mean() - overall_new_mae) / cluster_0['current_error'].mean() * 100:.1f}%")

# Visualize the splits
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Cluster 0 Sub-clustering Analysis', fontsize=16)

# 1. Scatter plot of sub-clusters
ax1 = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_0['hybrid_sub_cluster'].unique())))
for i, sub_cluster in enumerate(sorted(cluster_0['hybrid_sub_cluster'].unique())):
    sub_data = cluster_0[cluster_0['hybrid_sub_cluster'] == sub_cluster]
    ax1.scatter(sub_data['trip_days'], sub_data['miles'], 
               label=sub_cluster, alpha=0.6, color=colors[i])
ax1.set_xlabel('Trip Days')
ax1.set_ylabel('Miles')
ax1.set_title('Sub-clusters by Trip Days and Miles')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Error distribution by sub-cluster
ax2 = axes[0, 1]
sub_cluster_errors = []
sub_cluster_labels = []
for sub_cluster in sorted(cluster_0['hybrid_sub_cluster'].unique()):
    sub_data = cluster_0[cluster_0['hybrid_sub_cluster'] == sub_cluster]
    sub_cluster_errors.append(sub_data['current_error'].values)
    sub_cluster_labels.append(f"{sub_cluster}\n(n={len(sub_data)})")

ax2.boxplot(sub_cluster_errors, labels=sub_cluster_labels)
ax2.set_ylabel('Absolute Error ($)')
ax2.set_title('Error Distribution by Sub-cluster')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Receipt patterns
ax3 = axes[1, 0]
for i, sub_cluster in enumerate(sorted(cluster_0['hybrid_sub_cluster'].unique())):
    sub_data = cluster_0[cluster_0['hybrid_sub_cluster'] == sub_cluster]
    ax3.scatter(sub_data['receipts'], sub_data['current_error'], 
               label=sub_cluster, alpha=0.6, color=colors[i])
ax3.set_xlabel('Receipts ($)')
ax3.set_ylabel('Absolute Error ($)')
ax3.set_title('Error vs Receipts by Sub-cluster')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Sub-cluster sizes
ax4 = axes[1, 1]
sub_cluster_sizes = cluster_0['hybrid_sub_cluster'].value_counts()
ax4.pie(sub_cluster_sizes.values, labels=sub_cluster_sizes.index, autopct='%1.1f%%')
ax4.set_title('Sub-cluster Distribution')

plt.tight_layout()
plt.savefig('cluster_0_splitting_analysis.png', dpi=150, bbox_inches='tight')
print("\n\nSaved visualization to cluster_0_splitting_analysis.png")

# Save the sub-cluster models to a file
import json
model_config = {
    'sub_clusters': sub_cluster_models,
    'assignment_rules': {
        '0a_short': 'trip_days between 2 and 4',
        '0b_low_efficiency': 'miles_per_day < 50 and trip_days >= 5',
        '0c_long': 'trip_days >= 9',
        '0d_high_receipt': 'receipts > 1500 and trip_days >= 5',
        '0e_standard': 'default (everything else)'
    }
}

with open('cluster_0_sub_models.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print("\nSaved sub-cluster models to cluster_0_sub_models.json")

plt.show() 