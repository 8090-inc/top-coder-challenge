"""
Export Cluster Model Parameters
Saves cluster centroids and formulas for use in calculate_reimbursement v0.5
"""

import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

print("=" * 80)
print("EXPORTING CLUSTER MODEL PARAMETERS")
print("=" * 80)

# Load clustered data
df = pd.read_csv(PROCESSED_DATA_DIR / 'public_cases_with_clusters.csv')

# Add derived features if not present
if 'output_per_day' not in df.columns:
    df['output_per_day'] = df['expected_output'] / df['trip_days']

# Prepare feature columns (same as used in clustering)
feature_columns = [
    'trip_days', 'miles', 'receipts',
    'miles_per_day', 'receipts_per_day',
    'receipt_coverage', 'output_per_day'
]

# Prepare data
X = df[feature_columns].fillna(0)
X = X.replace([np.inf, -np.inf], 0)

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Refit K-means to get centroids
print("\nRefitting K-means clustering...")
kmeans = KMeans(n_clusters=6, random_state=42, n_init=20)
kmeans.fit(X_scaled)

# Get cluster assignments
df['cluster_check'] = kmeans.predict(X_scaled)

# Verify it matches our original clustering
if (df['cluster'] == df['cluster_check']).mean() < 0.95:
    print("WARNING: Cluster assignments don't match original!")

# Export cluster parameters
cluster_params = {
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'feature_columns': feature_columns,
    'centroids': kmeans.cluster_centers_.tolist(),
    'cluster_models': {}
}

# For each cluster, fit the best model
for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    
    print(f"\nCluster {cluster_id}: {len(cluster_data)} cases")
    
    if cluster_id == 4:  # Outlier case
        cluster_params['cluster_models'][cluster_id] = {
            'type': 'fixed',
            'value': 364.51
        }
        continue
    
    # Prepare features for prediction
    X_cluster = cluster_data[['trip_days', 'miles', 'receipts']].values
    y_cluster = cluster_data['expected_output'].values
    
    # Fit appropriate model based on our findings
    if cluster_id == 0:  # Standard multi-day - use linear
        model = LinearRegression()
        model.fit(X_cluster, y_cluster)
        cluster_params['cluster_models'][cluster_id] = {
            'type': 'linear',
            'intercept': float(model.intercept_),
            'coef': model.coef_.tolist()
        }
        print(f"  Linear model: {model.intercept_:.2f} + {model.coef_[0]:.2f}*days + {model.coef_[1]:.2f}*miles + {model.coef_[2]:.2f}*receipts")
        
    elif cluster_id == 5:  # Contains special profile
        # Handle special profile separately
        special_mask = (
            (cluster_data['trip_days'].between(7, 8)) &
            (cluster_data['miles'].between(900, 1200)) &
            (cluster_data['receipts'].between(1000, 1200))
        )
        special_cases = cluster_data[special_mask]
        
        # For special cases, use step function based on receipts
        if len(special_cases) > 0:
            # Define receipt bins and their outputs
            receipt_bins = [(1000, 1050, 2047),  # avg of cases in this range
                          (1050, 1100, 2073),
                          (1100, 1150, 2120),
                          (1150, 1200, 2280)]
            
            cluster_params['cluster_models'][cluster_id] = {
                'type': 'special_with_tree',
                'special_criteria': {
                    'trip_days': [7, 8],
                    'miles': [900, 1200],
                    'receipts': [1000, 1200]
                },
                'special_bins': receipt_bins
            }
        
        # Also fit a decision tree for non-special cases
        dt = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt.fit(X_cluster, y_cluster)
        
        # Export decision tree (simplified)
        cluster_params['cluster_models'][cluster_id]['tree_model'] = 'decision_tree_c5.pkl'
        with open(DATA_DIR / 'decision_tree_c5.pkl', 'wb') as f:
            pickle.dump(dt, f)
        
        print(f"  Special profile with decision tree fallback")
        
    else:  # Clusters 1, 2, 3 - use decision trees
        dt = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt.fit(X_cluster, y_cluster)
        
        # Export decision tree
        tree_file = f'decision_tree_c{cluster_id}.pkl'
        cluster_params['cluster_models'][cluster_id] = {
            'type': 'decision_tree',
            'model_file': tree_file
        }
        
        with open(DATA_DIR / tree_file, 'wb') as f:
            pickle.dump(dt, f)
        
        print(f"  Decision tree saved to {tree_file}")

# Save cluster parameters as JSON
output_file = DATA_DIR / 'cluster_model_params.json'
with open(output_file, 'w') as f:
    json.dump(cluster_params, f, indent=2)

print(f"\nCluster parameters saved to: {output_file}")

# Also create a simplified version for embedding in calculate_reimbursement.py
simplified_params = {
    'scaler': {
        'mean': cluster_params['scaler_mean'],
        'scale': cluster_params['scaler_scale']
    },
    'centroids': cluster_params['centroids'],
    'models': {}
}

# Simplify model parameters
for cluster_id in range(6):
    model_info = cluster_params['cluster_models'][cluster_id]
    
    if model_info['type'] == 'linear':
        simplified_params['models'][cluster_id] = {
            'type': 'linear',
            'formula': f"{model_info['intercept']:.2f} + {model_info['coef'][0]:.2f}*days + {model_info['coef'][1]:.2f}*miles + {model_info['coef'][2]:.2f}*receipts"
        }
    elif model_info['type'] == 'fixed':
        simplified_params['models'][cluster_id] = {
            'type': 'fixed',
            'value': model_info['value']
        }
    elif model_info['type'] == 'special_with_tree':
        simplified_params['models'][cluster_id] = {
            'type': 'special',
            'note': 'Complex rules for cluster 5'
        }
    else:
        simplified_params['models'][cluster_id] = {
            'type': 'tree',
            'note': 'Decision tree required'
        }

print("\n" + "=" * 80)
print("SIMPLIFIED MODEL SUMMARY")
print("=" * 80)

for cluster_id, model in simplified_params['models'].items():
    print(f"\nCluster {cluster_id}: {model['type']}")
    if 'formula' in model:
        print(f"  Formula: {model['formula']}")
    elif 'value' in model:
        print(f"  Fixed value: ${model['value']:.2f}")
    else:
        print(f"  {model['note']}")

print("\n" + "=" * 80)
print("EXPORT COMPLETE")
print("=" * 80) 