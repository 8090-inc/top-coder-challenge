#!/usr/bin/env python3
"""
Train V6 Cluster Residual Ensemble Model
Includes 5-fold CV on 850 train samples and locked 150 blind test
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from v6_cluster_residual_ensemble import ClusterResidualEnsemble

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("Training V6 Cluster Residual Ensemble Model")
print("=" * 60)

# Load data
print("\nLoading data...")
with open('../data/raw/public_cases.json', 'r') as f:
    cases = json.load(f)

# Convert to DataFrame
data = []
for case in cases:
    data.append({
        'trip_days': case['input']['trip_duration_days'],
        'miles_traveled': case['input']['miles_traveled'],
        'total_receipts_amount': case['input']['total_receipts_amount'],
        'expected_reimbursement': case['expected_output']
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} cases")

# Create train/blind split (850/150)
print("\nCreating train/blind split...")
train_df, blind_df = train_test_split(df, test_size=150, random_state=42)
print(f"Train set: {len(train_df)} cases")
print(f"Blind test set: {len(blind_df)} cases")

# Run 5-fold cross-validation on training set
print("\n" + "=" * 60)
print("PHASE 1: Cross-Validation on Training Set")
print("=" * 60)

model = ClusterResidualEnsemble()
cv_avg_mae, cv_std_mae, fold_maes = model.cross_validate(train_df, n_splits=5)

# Train final model on full training set
print("\n" + "=" * 60)
print("PHASE 2: Training Final Model on Full Training Set")
print("=" * 60)

final_model = ClusterResidualEnsemble()
train_mae = final_model.train(train_df)

# Evaluate on blind test set
print("\n" + "=" * 60)
print("PHASE 3: Blind Test Set Evaluation")
print("=" * 60)

blind_preds = final_model.predict_batch(blind_df)
blind_errors = blind_preds - blind_df['expected_reimbursement'].values
blind_abs_errors = np.abs(blind_errors)

blind_mae = np.mean(blind_abs_errors)
blind_max_error = np.max(blind_abs_errors)
blind_90th = np.percentile(blind_abs_errors, 90)
blind_95th = np.percentile(blind_abs_errors, 95)

print(f"\nBlind Test Results:")
print(f"  MAE: ${blind_mae:.2f}")
print(f"  Max Error: ${blind_max_error:.2f}")
print(f"  90th percentile: ${blind_90th:.2f}")
print(f"  95th percentile: ${blind_95th:.2f}")

# Show worst cases
worst_idx = np.argsort(blind_abs_errors)[-5:]
print(f"\nWorst 5 blind test cases:")
for idx in worst_idx:
    row = blind_df.iloc[idx]
    print(f"  Days={row['trip_days']}, Miles={row['miles_traveled']:.0f}, "
          f"Receipts=${row['total_receipts_amount']:.2f}")
    print(f"    Expected: ${row['expected_reimbursement']:.2f}, "
          f"Predicted: ${blind_preds[idx]:.2f}, "
          f"Error: ${blind_abs_errors[idx]:.2f}")

# Save model if performance is good
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nPerformance Summary:")
print(f"  Cross-validation MAE: ${cv_avg_mae:.2f} (±${cv_std_mae:.2f})")
print(f"  Training MAE: ${train_mae:.2f}")
print(f"  Blind test MAE: ${blind_mae:.2f}")
print(f"  Blind vs CV difference: ${blind_mae - cv_avg_mae:.2f}")

# Decision criteria
target_train_mae = 40.0
target_blind_mae = 45.0
max_error_threshold = 200.0

print(f"\nTarget criteria:")
print(f"  Train MAE < ${target_train_mae:.0f}: {'✓' if train_mae < target_train_mae else '✗'}")
print(f"  Blind MAE < ${target_blind_mae:.0f}: {'✓' if blind_mae < target_blind_mae else '✗'}")
print(f"  Max error < ${max_error_threshold:.0f}: {'✓' if blind_max_error < max_error_threshold else '✗'}")

if train_mae < target_train_mae and blind_mae < target_blind_mae and blind_max_error < max_error_threshold:
    print("\n✅ All criteria met! Saving model...")
    final_model.save('v6_cluster_residual_ensemble.pkl')
    
    # Also save performance metrics
    metrics = {
        'cv_mae': float(cv_avg_mae),
        'cv_std': float(cv_std_mae),
        'train_mae': float(train_mae),
        'blind_mae': float(blind_mae),
        'blind_max_error': float(blind_max_error),
        'blind_90th': float(blind_90th),
        'blind_95th': float(blind_95th),
        'train_size': len(train_df),
        'blind_size': len(blind_df)
    }
    
    with open('v6_performance_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Performance metrics saved to v6_performance_metrics.json")
else:
    print("\n⚠️ Performance targets not met. Consider implementing bullet #6 (caps) for clusters 1a & 2")
    
    # Analyze per-cluster performance
    print("\nPer-cluster training statistics:")
    for cluster_id, stats in final_model.training_stats.items():
        print(f"  Cluster {cluster_id}: n={stats['n_samples']}, "
              f"MAE=${stats['in_sample_mae']:.2f}, "
              f"bound=${stats['adaptive_bound']:.2f}")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60) 