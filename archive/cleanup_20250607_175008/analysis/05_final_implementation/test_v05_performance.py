"""
Test v0.5 Performance by Cluster
Analyze where the errors are coming from
"""

import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
sys.path.append(str(Path(__file__).parent.parent.parent))
import calculate_reimbursement as calc

print("=" * 80)
print("V0.5 PERFORMANCE ANALYSIS BY CLUSTER")
print("=" * 80)

# Load clustered data
df = pd.read_csv(PROCESSED_DATA_DIR / 'public_cases_with_clusters.csv')

# Calculate v0.5 predictions
print("\nCalculating v0.5 predictions...")
df['v05_predicted'] = df.apply(
    lambda row: calc.calculate_reimbursement(row['trip_days'], row['miles'], row['receipts'], version='v0.5'),
    axis=1
)

# Calculate errors
df['v05_error'] = df['v05_predicted'] - df['expected_output']
df['v05_abs_error'] = abs(df['v05_error'])

# Overall performance
print(f"\nOverall v0.5 MAE: ${df['v05_abs_error'].mean():.2f}")

# Performance by cluster
print("\n" + "=" * 60)
print("PERFORMANCE BY CLUSTER")
print("=" * 60)

for cluster_id in range(6):
    cluster_data = df[df['cluster'] == cluster_id]
    
    print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
    print(f"  MAE: ${cluster_data['v05_abs_error'].mean():.2f}")
    print(f"  Mean error: ${cluster_data['v05_error'].mean():.2f}")
    print(f"  Std error: ${cluster_data['v05_error'].std():.2f}")
    
    # Show worst predictions
    worst = cluster_data.nlargest(5, 'v05_abs_error')
    if len(worst) > 0:
        print("\n  Worst predictions:")
        for idx, row in worst.iterrows():
            print(f"    Days: {row['trip_days']:.0f}, Miles: {row['miles']:.0f}, "
                  f"Receipts: ${row['receipts']:.2f}")
            print(f"    Expected: ${row['expected_output']:.0f}, "
                  f"Predicted: ${row['v05_predicted']:.0f}, "
                  f"Error: ${row['v05_abs_error']:.0f}")

# Check cluster assignment accuracy
print("\n" + "=" * 60)
print("CLUSTER ASSIGNMENT CHECK")
print("=" * 60)

# For each case, check which cluster v0.5 would assign it to
def get_v05_cluster(trip_days, miles, receipts):
    """Extract cluster assignment from v0.5 logic"""
    # Calculate derived features
    miles_per_day = miles / trip_days if trip_days > 0 else 0
    receipts_per_day = receipts / trip_days if trip_days > 0 else 0
    receipt_coverage = 1.0
    output_per_day = 200
    
    features = [trip_days, miles, receipts, miles_per_day, receipts_per_day, receipt_coverage, output_per_day]
    
    # Scaler parameters
    scaler_mean = [7.043, 597.41374, 1211.0568700000001, 147.02619530669332, 
                   285.7060807000777, 2.801597213397003, 284.7120321275669]
    scaler_scale = [3.9241751999624075, 351.124095966102, 742.4826603738993, 193.7236752036585,
                    381.51689150974863, 10.153163246773477, 268.43394376874403]
    
    # Scale features
    scaled_features = []
    for i, (feat, mean, scale) in enumerate(zip(features, scaler_mean, scaler_scale)):
        if scale > 0:
            scaled_features.append((feat - mean) / scale)
        else:
            scaled_features.append(0)
    
    # Centroids
    centroids = [
        [0.1866283439250101, -0.9967064552506947, -0.7474365546997919, -0.5569162609593883, 
         -0.5034396321321946, 0.09950439973002874, -0.5273182311434319],
        [-1.5348448254954827, 0.4564946178329905, 0.5543886099544929, 3.0925172365405995, 
         3.4426143338043045, -0.19235370419136444, 3.570176909884608],
        [0.8714149750618587, 0.3845216918453981, 0.8276242210045852, -0.37621620555194757, 
         -0.2616218096891219, -0.16992681039252977, -0.39186643124293],
        [-0.9295332081059077, -0.10675734772962193, 0.7021487155222411, 0.2573874915645624, 
         0.7561263547962562, -0.19134009213310563, 0.6916318602105594],
        [-1.03028019748933, -1.4365682839627638, -1.629178611916474, -0.5989262550626138, 
         -0.7476280964599334, 25.006550147401477, -0.6080032074290795],
        [-0.3384212189613996, 0.7681918137606423, -0.888351994366373, 0.31893173109174644, 
         -0.413384884548907, 0.19331835187911464, -0.17445874232854566]
    ]
    
    # Find nearest centroid
    min_distance = float('inf')
    cluster_id = 0
    
    for i, centroid in enumerate(centroids):
        distance = sum((a - b) ** 2 for a, b in zip(scaled_features, centroid))
        if distance < min_distance:
            min_distance = distance
            cluster_id = i
    
    return cluster_id

# Check cluster assignments
print("\nChecking cluster assignments...")
df['v05_cluster'] = df.apply(
    lambda row: get_v05_cluster(row['trip_days'], row['miles'], row['receipts']),
    axis=1
)

# Create confusion matrix
confusion_matrix = pd.crosstab(df['cluster'], df['v05_cluster'], margins=True)
print("\nCluster Assignment Confusion Matrix:")
print("(Rows: True cluster, Columns: v0.5 assigned cluster)")
print(confusion_matrix)

# Calculate assignment accuracy
correct_assignments = (df['cluster'] == df['v05_cluster']).sum()
accuracy = correct_assignments / len(df) * 100
print(f"\nCluster assignment accuracy: {accuracy:.1f}%")

# Check special profile cases
print("\n" + "=" * 60)
print("SPECIAL PROFILE CASES CHECK")
print("=" * 60)

special_cases = df[
    (df['trip_days'].between(7, 8)) &
    (df['miles'].between(900, 1200)) &
    (df['receipts'].between(1000, 1200))
]

print(f"\nFound {len(special_cases)} special profile cases")
print("\nSpecial case predictions:")
for idx, row in special_cases.iterrows():
    print(f"  Days: {row['trip_days']:.0f}, Miles: {row['miles']:.0f}, "
          f"Receipts: ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected_output']:.0f}, "
          f"Predicted: ${row['v05_predicted']:.0f}, "
          f"Error: ${row['v05_abs_error']:.0f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80) 