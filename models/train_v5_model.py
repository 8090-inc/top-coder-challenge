#!/usr/bin/env python3
"""
Train the V5 Practical Ensemble Model

This uses all 1000 public cases for training to maximize learning.
The model includes conservative corrections to avoid overfitting.
"""

import pandas as pd
import sys
sys.path.append('..')
from v5_practical_ensemble import PracticalEnsembleModel

# Load all training data
print("Loading training data...")
train_df = pd.read_csv('../data/train.csv')
print(f"Loaded {len(train_df)} training cases")

# Create and train model
print("\n" + "="*60)
model = PracticalEnsembleModel(max_correction=100)
final_mae = model.train(train_df)

# Save the trained model
print("\n" + "="*60)
print("Saving trained model...")
model.save('v5_practical_ensemble.pkl')

# Summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Training samples: {model.training_stats['n_samples']}")
print(f"Final MAE: ${final_mae:.2f}")
print(f"Improvement over baseline: {(114.79 - final_mae) / 114.79 * 100:.1f}%")
print(f"\nModel saved to: v5_practical_ensemble.pkl")
print("\nNext steps:")
print("1. The model is ready for production use")
print("2. Focus on discovering the cents hash for exact matches")
print("3. Consider gathering more training data if possible") 