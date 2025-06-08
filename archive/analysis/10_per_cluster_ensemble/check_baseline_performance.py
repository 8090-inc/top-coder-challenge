#!/usr/bin/env python3
"""
Check baseline rule engine performance to understand the fundamental challenge
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append('../..')
from models.cluster_models_optimized import calculate_reimbursement_v3 as predict_reimbursement

# Load data
train_df = pd.read_csv('../../data/train.csv')

# Same split as before
train_main, train_blind = train_test_split(train_df, test_size=0.15, random_state=42)

print("=== BASELINE RULE ENGINE PERFORMANCE ===\n")

# Evaluate on both sets
for name, df in [('Training', train_main), ('Blind Test', train_blind)]:
    predictions = []
    for _, row in df.iterrows():
        pred = predict_reimbursement(
            row['trip_days'],
            row['miles_traveled'],
            row['total_receipts_amount']
        )
        predictions.append(pred)
    
    df['predicted'] = predictions
    df['error'] = abs(df['expected_reimbursement'] - df['predicted'])
    
    mae = df['error'].mean()
    max_error = df['error'].max()
    
    print(f"{name} Set ({len(df)} samples):")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Max Error: ${max_error:.2f}")
    print(f"  90th percentile: ${np.percentile(df['error'], 90):.2f}")
    
    # Show worst cases
    worst = df.nlargest(5, 'error')[['trip_days', 'miles_traveled', 'total_receipts_amount', 
                                      'expected_reimbursement', 'predicted', 'error']]
    print(f"\n  Worst 5 cases:")
    for _, row in worst.iterrows():
        print(f"    {row['trip_days']}d, {row['miles_traveled']:.0f}mi, ${row['total_receipts_amount']:.2f} -> "
              f"Expected ${row['expected_reimbursement']:.2f}, Got ${row['predicted']:.2f}, Error ${row['error']:.2f}")
    print()

# Check residual magnitudes
print("\n=== RESIDUAL ANALYSIS ===")
train_residuals = train_main['expected_reimbursement'] - train_main['predicted']
blind_residuals = train_blind['expected_reimbursement'] - train_blind['predicted']

print(f"\nTraining residuals:")
print(f"  Mean: ${train_residuals.mean():.2f}")
print(f"  Std: ${train_residuals.std():.2f}")
print(f"  Range: ${train_residuals.min():.2f} to ${train_residuals.max():.2f}")

print(f"\nBlind residuals:")
print(f"  Mean: ${blind_residuals.mean():.2f}")
print(f"  Std: ${blind_residuals.std():.2f}")
print(f"  Range: ${blind_residuals.min():.2f} to ${blind_residuals.max():.2f}")

# Check if blind set has different distribution
print("\n=== DISTRIBUTION COMPARISON ===")
for col in ['trip_days', 'miles_traveled', 'total_receipts_amount']:
    train_mean = train_main[col].mean()
    blind_mean = train_blind[col].mean()
    diff_pct = (blind_mean - train_mean) / train_mean * 100
    print(f"{col}: Train mean={train_mean:.1f}, Blind mean={blind_mean:.1f} ({diff_pct:+.1f}%)") 