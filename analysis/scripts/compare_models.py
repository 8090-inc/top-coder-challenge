"""
Compare different model versions to identify strengths and weaknesses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# Load predictions from both models
v01_df = pd.read_csv('data/raw/public_cases_predictions_v0.1.csv')
v02_df = pd.read_csv('data/raw/public_cases_predictions_v0.2.csv')

# Merge on common columns
df = v01_df[['trip_days', 'miles', 'receipts', 'expected_output']].copy()
df['pred_v01'] = v01_df['predicted']
df['pred_v02'] = v02_df['predicted']
df['error_v01'] = abs(df['pred_v01'] - df['expected_output'])
df['error_v02'] = abs(df['pred_v02'] - df['expected_output'])
df['v02_better'] = df['error_v02'] < df['error_v01']

print("MODEL COMPARISON ANALYSIS")
print("=" * 60)

# Overall statistics
print("\nOVERALL PERFORMANCE:")
print(f"v0.1 MAE: ${df['error_v01'].mean():.2f}")
print(f"v0.2 MAE: ${df['error_v02'].mean():.2f}")
print(f"Cases where v0.2 is better: {df['v02_better'].sum()} / {len(df)} ({df['v02_better'].mean():.1%})")

# Analyze by receipt ranges
print("\nPERFORMANCE BY RECEIPT RANGE:")
receipt_ranges = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 3000)]

for low, high in receipt_ranges:
    mask = (df['receipts'] >= low) & (df['receipts'] < high)
    subset = df[mask]
    if len(subset) > 0:
        v01_mae = subset['error_v01'].mean()
        v02_mae = subset['error_v02'].mean()
        v02_wins = subset['v02_better'].mean()
        print(f"\n${low}-${high} receipts ({len(subset)} cases):")
        print(f"  v0.1 MAE: ${v01_mae:.2f}")
        print(f"  v0.2 MAE: ${v02_mae:.2f}")
        print(f"  v0.2 wins: {v02_wins:.1%}")
        print(f"  Better model: {'v0.2' if v02_mae < v01_mae else 'v0.1'}")

# Find patterns where v0.2 excels
print("\n\nCASES WHERE v0.2 EXCELS (error < $50):")
v02_excellent = df[df['error_v02'] < 50]
print(f"Total cases: {len(v02_excellent)}")
print("\nReceipt distribution of v0.2 excellent cases:")
print(v02_excellent['receipts'].describe())

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Comparison: v0.1 (Linear) vs v0.2 (Inverted Coverage)', fontsize=16)

# 1. Error comparison by receipt amount
ax1 = axes[0, 0]
ax1.scatter(df['receipts'], df['error_v01'], alpha=0.5, label='v0.1 errors', s=20)
ax1.scatter(df['receipts'], df['error_v02'], alpha=0.5, label='v0.2 errors', s=20)
ax1.set_xlabel('Receipt Amount ($)')
ax1.set_ylabel('Absolute Error ($)')
ax1.set_title('Model Errors by Receipt Amount')
ax1.legend()

# 2. Where each model wins
ax2 = axes[0, 1]
v01_wins = df[~df['v02_better']]
v02_wins = df[df['v02_better']]
ax2.scatter(v01_wins['receipts'], v01_wins['expected_output'], 
           alpha=0.5, label=f'v0.1 better (n={len(v01_wins)})', s=30)
ax2.scatter(v02_wins['receipts'], v02_wins['expected_output'], 
           alpha=0.5, label=f'v0.2 better (n={len(v02_wins)})', s=30)
ax2.set_xlabel('Receipt Amount ($)')
ax2.set_ylabel('Expected Output ($)')
ax2.set_title('Cases Where Each Model Performs Better')
ax2.legend()

# 3. Error improvement heatmap
ax3 = axes[1, 0]
# Create 2D histogram
receipt_bins = np.linspace(0, 2500, 25)
output_bins = np.linspace(0, 2500, 25)
improvement = df['error_v01'] - df['error_v02']  # Positive means v0.2 is better

h, xedges, yedges = np.histogram2d(df['receipts'], df['expected_output'], 
                                   bins=[receipt_bins, output_bins], 
                                   weights=improvement)
counts, _, _ = np.histogram2d(df['receipts'], df['expected_output'], 
                             bins=[receipt_bins, output_bins])

# Average improvement per bin
h_avg = np.divide(h, counts, out=np.zeros_like(h), where=counts!=0)

im = ax3.imshow(h_avg.T, origin='lower', aspect='auto', cmap='RdBu',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax3.set_xlabel('Receipt Amount ($)')
ax3.set_ylabel('Expected Output ($)')
ax3.set_title('Average Error Improvement (Blue = v0.2 better)')
plt.colorbar(im, ax=ax3, label='Error Improvement ($)')

# 4. Residual patterns
ax4 = axes[1, 1]
# Calculate residuals for both models
residual_v01 = df['pred_v01'] - df['expected_output']
residual_v02 = df['pred_v02'] - df['expected_output']

ax4.scatter(df['receipts'], residual_v01, alpha=0.5, label='v0.1', s=20)
ax4.scatter(df['receipts'], residual_v02, alpha=0.5, label='v0.2', s=20)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.set_xlabel('Receipt Amount ($)')
ax4.set_ylabel('Prediction Residual ($)')
ax4.set_title('Model Residuals by Receipt Amount')
ax4.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_comparison_v01_v02.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to {FIGURES_DIR / 'model_comparison_v01_v02.png'}")

# Identify hybrid strategy
print("\n\nHYBRID MODEL STRATEGY:")
print("Based on the analysis, a hybrid approach might work:")
print("- Use inverted coverage for receipts in certain ranges")
print("- Use linear model as base with adjustments")
print("- Key insight: v0.2 excels around $1000-1200 receipts")

# Save detailed comparison
comparison_df = df[['trip_days', 'miles', 'receipts', 'expected_output', 
                    'pred_v01', 'pred_v02', 'error_v01', 'error_v02', 'v02_better']]
comparison_df.to_csv(REPORTS_DIR / 'model_comparison_v01_v02.csv', index=False)
print(f"\nDetailed comparison saved to {REPORTS_DIR / 'model_comparison_v01_v02.csv'}") 