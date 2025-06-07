"""
Clustering Analysis to Find Multiple Calculation Paths
Based on Kevin's observation of 6 different paths
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
print("CLUSTERING ANALYSIS - FINDING CALCULATION PATHS")
print("=" * 80)

# Load data and predictions
print("\nLoading data...")
with open(PUBLIC_CASES_PATH, "r") as f:
    data = json.load(f)
df = pd.json_normalize(data)
df.columns = ['expected_output', 'trip_days', 'miles', 'receipts']

# Add v0.4 predictions and residuals
df['predicted'] = df.apply(
    lambda row: calc.calculate_reimbursement(row['trip_days'], row['miles'], row['receipts'], version='v0.4'),
    axis=1
)
df['residual'] = df['expected_output'] - df['predicted']
df['abs_residual'] = abs(df['residual'])
df['pct_error'] = (df['abs_residual'] / df['expected_output']) * 100

# Add derived features
df['miles_per_day'] = df['miles'] / df['trip_days']
df['receipts_per_day'] = df['receipts'] / df['trip_days']
df['receipt_coverage'] = df['expected_output'] / df['receipts']
df['output_per_day'] = df['expected_output'] / df['trip_days']
df['receipt_cents'] = (df['receipts'] * 100) % 100

print(f"Loaded {len(df)} cases")
print(f"Current model MAE: ${df['abs_residual'].mean():.2f}")

# 1. ANALYZE RESIDUAL PATTERNS
print("\n1. RESIDUAL PATTERN ANALYSIS")
print("-" * 40)

# Find systematic errors
high_positive_residuals = df[df['residual'] > 200]
high_negative_residuals = df[df['residual'] < -200]

print(f"High positive residuals (under-predicted): {len(high_positive_residuals)} cases")
print(f"High negative residuals (over-predicted): {len(high_negative_residuals)} cases")

# 2. FEATURE ENGINEERING FOR CLUSTERING
print("\n2. PREPARING FEATURES FOR CLUSTERING")
print("-" * 40)

# Select features for clustering
feature_columns = [
    'trip_days', 'miles', 'receipts',
    'miles_per_day', 'receipts_per_day',
    'receipt_coverage', 'output_per_day'
]

# Also create a version with residuals to find error patterns
feature_columns_with_residual = feature_columns + ['residual', 'pct_error']

# Prepare data
X = df[feature_columns].fillna(0)
X_with_residual = df[feature_columns_with_residual].fillna(0)

# Handle infinite values
X = X.replace([np.inf, -np.inf], 0)
X_with_residual = X_with_residual.replace([np.inf, -np.inf], 0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_residual_scaled = scaler.fit_transform(X_with_residual)

print(f"Features prepared: {feature_columns}")

# 3. DETERMINE OPTIMAL NUMBER OF CLUSTERS
print("\n3. FINDING OPTIMAL NUMBER OF CLUSTERS")
print("-" * 40)

# Test different numbers of clusters
inertias = []
silhouette_scores = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Find elbow point
print("Silhouette scores by number of clusters:")
for k, score in zip(K_range, silhouette_scores):
    print(f"  k={k}: {score:.3f}")

# Kevin mentioned 6 paths, let's test around that
optimal_k = 6
print(f"\nUsing k={optimal_k} (based on Kevin's observation)")

# 4. PERFORM CLUSTERING
print("\n4. PERFORMING K-MEANS CLUSTERING")
print("-" * 40)

# K-means with 6 clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print("\nCluster sizes:")
cluster_sizes = df['cluster'].value_counts().sort_index()
for cluster, size in cluster_sizes.items():
    print(f"  Cluster {cluster}: {size} cases")

# 5. CHARACTERIZE EACH CLUSTER
print("\n5. CLUSTER CHARACTERISTICS")
print("-" * 40)

for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"\nCLUSTER {cluster_id} ({len(cluster_data)} cases):")
    
    # Basic stats
    print(f"  Trip days: {cluster_data['trip_days'].mean():.1f} ± {cluster_data['trip_days'].std():.1f}")
    print(f"  Miles: {cluster_data['miles'].mean():.0f} ± {cluster_data['miles'].std():.0f}")
    print(f"  Receipts: ${cluster_data['receipts'].mean():.0f} ± {cluster_data['receipts'].std():.0f}")
    print(f"  Output: ${cluster_data['expected_output'].mean():.0f} ± {cluster_data['expected_output'].std():.0f}")
    
    # Model performance
    print(f"  Model MAE: ${cluster_data['abs_residual'].mean():.2f}")
    print(f"  Model MAPE: {cluster_data['pct_error'].mean():.1f}%")
    
    # Check if this matches our known special profile
    if (cluster_data['trip_days'].between(7, 8).mean() > 0.8 and
        cluster_data['miles'].between(900, 1200).mean() > 0.8 and
        cluster_data['receipts'].between(1000, 1200).mean() > 0.8):
        print("  *** MATCHES KNOWN SPECIAL PROFILE ***")

# 6. FIND PATTERNS IN HIGH-ERROR CLUSTERS
print("\n6. ANALYZING HIGH-ERROR PATTERNS")
print("-" * 40)

# For each cluster, find the worst predictions
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    cluster_mae = cluster_data['abs_residual'].mean()
    
    if cluster_mae > df['abs_residual'].mean() * 1.5:  # 50% worse than average
        print(f"\nHIGH-ERROR CLUSTER {cluster_id}:")
        worst_cases = cluster_data.nlargest(5, 'abs_residual')
        
        # Look for patterns
        print("  Common characteristics:")
        
        # Check receipt endings
        special_endings = cluster_data['receipt_cents'].isin([49, 99]).mean()
        if special_endings > 0.2:
            print(f"    - {special_endings:.1%} have .49/.99 endings (vs {df['receipt_cents'].isin([49, 99]).mean():.1%} overall)")
        
        # Check trip length patterns
        trip_mode = cluster_data['trip_days'].mode().values[0] if len(cluster_data['trip_days'].mode()) > 0 else None
        if trip_mode:
            print(f"    - Most common trip length: {trip_mode} days")
        
        # Check receipt ranges
        low_receipts = (cluster_data['receipts'] < 50).mean()
        if low_receipts > 0.3:
            print(f"    - {low_receipts:.1%} have receipts < $50")

# 7. DBSCAN FOR OUTLIER DETECTION
print("\n7. DBSCAN OUTLIER DETECTION")
print("-" * 40)

# Use DBSCAN to find outliers
dbscan = DBSCAN(eps=1.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

outliers = df[df['dbscan_cluster'] == -1]
print(f"DBSCAN found {len(outliers)} outliers")

if len(outliers) > 0:
    print("\nOutlier characteristics:")
    print(f"  Mean error: ${outliers['abs_residual'].mean():.2f} vs ${df['abs_residual'].mean():.2f} overall")
    print(f"  Receipt range: ${outliers['receipts'].min():.2f} - ${outliers['receipts'].max():.2f}")

# 8. VISUALIZATION
print("\n8. CREATING VISUALIZATIONS")
print("-" * 40)

fig = plt.figure(figsize=(20, 16))

# 1. Elbow plot for k-means
ax1 = plt.subplot(3, 3, 1)
ax1.plot(K_range, inertias, 'b-', marker='o')
ax1.axvline(x=6, color='r', linestyle='--', label='k=6 (Kevin\'s observation)')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Inertia')
ax1.set_title('K-Means Elbow Plot')
ax1.legend()

# 2. Silhouette scores
ax2 = plt.subplot(3, 3, 2)
ax2.plot(K_range, silhouette_scores, 'g-', marker='o')
ax2.axvline(x=6, color='r', linestyle='--', label='k=6')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Scores by k')
ax2.legend()

# 3. PCA visualization of clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ax3 = plt.subplot(3, 3, 3)
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='tab10', alpha=0.6)
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax3.set_title('Clusters in PCA Space')
plt.colorbar(scatter, ax=ax3, label='Cluster')

# 4-6. Cluster characteristics plots
for i, feature in enumerate(['trip_days', 'miles', 'receipts']):
    ax = plt.subplot(3, 3, 4 + i)
    df.boxplot(column=feature, by='cluster', ax=ax)
    ax.set_title(f'{feature.replace("_", " ").title()} by Cluster')
    ax.set_xlabel('Cluster')

# 7. Error by cluster
ax7 = plt.subplot(3, 3, 7)
df.boxplot(column='abs_residual', by='cluster', ax=ax7)
ax7.set_title('Model Error by Cluster')
ax7.set_xlabel('Cluster')
ax7.set_ylabel('Absolute Error ($)')

# 8. Receipt coverage by cluster
ax8 = plt.subplot(3, 3, 8)
# Cap extreme values for visualization
receipt_coverage_capped = df['receipt_coverage'].clip(upper=10)
df_viz = df.copy()
df_viz['receipt_coverage_capped'] = receipt_coverage_capped
df_viz.boxplot(column='receipt_coverage_capped', by='cluster', ax=ax8)
ax8.set_title('Receipt Coverage by Cluster (capped at 10x)')
ax8.set_xlabel('Cluster')
ax8.set_ylabel('Coverage Ratio')

# 9. Cluster sizes
ax9 = plt.subplot(3, 3, 9)
cluster_sizes.plot(kind='bar', ax=ax9)
ax9.set_xlabel('Cluster')
ax9.set_ylabel('Number of Cases')
ax9.set_title('Cluster Sizes')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to '{FIGURES_DIR / 'clustering_analysis.png'}'")

# 9. IDENTIFY POTENTIAL NEW CALCULATION PATHS
print("\n9. POTENTIAL NEW CALCULATION PATHS")
print("-" * 40)

# Look for clusters with distinct patterns
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    
    # Check if this cluster has a distinct profile
    trip_day_concentration = cluster_data['trip_days'].value_counts().head(1).values[0] / len(cluster_data)
    
    if trip_day_concentration > 0.3:  # 30% have same trip length
        common_days = cluster_data['trip_days'].mode().values[0]
        subset = cluster_data[cluster_data['trip_days'] == common_days]
        
        print(f"\nCluster {cluster_id} - Potential path for {common_days}-day trips:")
        print(f"  Cases: {len(subset)}")
        print(f"  Miles range: {subset['miles'].min():.0f} - {subset['miles'].max():.0f}")
        print(f"  Receipt range: ${subset['receipts'].min():.0f} - ${subset['receipts'].max():.0f}")
        print(f"  Output range: ${subset['expected_output'].min():.0f} - ${subset['expected_output'].max():.0f}")
        print(f"  Current model error: ${subset['abs_residual'].mean():.2f}")

# 10. SAVE RESULTS
print("\n10. SAVING CLUSTER ASSIGNMENTS")
print("-" * 40)

# Save enriched dataset with cluster assignments
output_df = df[['trip_days', 'miles', 'receipts', 'expected_output', 
                 'predicted', 'residual', 'cluster', 'miles_per_day', 
                 'receipts_per_day', 'receipt_coverage']]
output_df.to_csv(PROCESSED_DATA_DIR / 'public_cases_with_clusters.csv', index=False)
print(f"Saved cluster assignments to '{PROCESSED_DATA_DIR / 'public_cases_with_clusters.csv'}'")

# Save cluster summaries
cluster_summary = []
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    summary = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_trip_days': cluster_data['trip_days'].mean(),
        'avg_miles': cluster_data['miles'].mean(),
        'avg_receipts': cluster_data['receipts'].mean(),
        'avg_output': cluster_data['expected_output'].mean(),
        'model_mae': cluster_data['abs_residual'].mean(),
        'common_trip_days': cluster_data['trip_days'].mode().values[0] if len(cluster_data['trip_days'].mode()) > 0 else None
    }
    cluster_summary.append(summary)

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df.to_csv(REPORTS_DIR / 'cluster_summary.csv', index=False)
print(f"Saved cluster summary to '{REPORTS_DIR / 'cluster_summary.csv'}'")

print("\n" + "=" * 80)
print("CLUSTERING ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nFound {optimal_k} distinct clusters")
print("Check visualizations and summaries for potential new calculation paths")
print("\nNext steps:")
print("1. Analyze each cluster for unique calculation patterns")
print("2. Test cluster-specific models")
print("3. Look for systematic differences in high-error clusters") 