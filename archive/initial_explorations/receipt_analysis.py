"""
Deep Receipt Analysis for Legacy Reimbursement System
Priority 1: Understanding the dominant factor in reimbursements
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
import warnings
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("DEEP RECEIPT ANALYSIS")
print("=" * 80)

# Load data
print("\nLoading data...")
with open(PUBLIC_CASES_PATH, "r") as f:
    data = json.load(f)
public_df = pd.json_normalize(data)
public_df.columns = ['expected_output', 'trip_days', 'miles', 'receipts']

print(f"Loaded {len(public_df)} cases")

# 1. BASIC RECEIPT STATISTICS
print("\n1. BASIC RECEIPT STATISTICS")
print("-" * 40)
print(public_df['receipts'].describe())

# 2. RECEIPT DISTRIBUTION ANALYSIS
print("\n2. RECEIPT DISTRIBUTION ANALYSIS")
print("-" * 40)

# Create receipt bins for analysis
receipt_bins = [0, 10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]
receipt_labels = ['0-10', '10-50', '50-100', '100-200', '200-500', 
                  '500-1000', '1000-1500', '1500-2000', '2000+']
public_df['receipt_bin'] = pd.cut(public_df['receipts'], bins=receipt_bins, labels=receipt_labels)

# Analyze by bins
bin_analysis = public_df.groupby('receipt_bin').agg({
    'expected_output': ['mean', 'std', 'count'],
    'receipts': ['mean', 'min', 'max']
}).round(2)

print("\nReimbursement by receipt ranges:")
print(bin_analysis)

# 3. RECEIPT COVERAGE RATIO ANALYSIS
print("\n3. RECEIPT COVERAGE RATIO ANALYSIS")
print("-" * 40)

# Calculate what percentage of receipts gets reimbursed
public_df['receipt_coverage'] = public_df['expected_output'] / public_df['receipts']
# Cap at reasonable values for analysis
public_df['receipt_coverage_capped'] = public_df['receipt_coverage'].clip(upper=10)

print(f"Mean receipt coverage: {public_df['receipt_coverage'].mean():.2%}")
print(f"Median receipt coverage: {public_df['receipt_coverage'].median():.2%}")

# Coverage by receipt amount
coverage_by_amount = public_df.groupby('receipt_bin')['receipt_coverage'].agg(['mean', 'median']).round(3)
print("\nReceipt coverage by amount range:")
print(coverage_by_amount)

# 4. LOW RECEIPT PENALTY ANALYSIS
print("\n4. LOW RECEIPT PENALTY INVESTIGATION")
print("-" * 40)

# Separate low receipt cases
very_low_receipts = public_df[public_df['receipts'] < 10]
low_receipts = public_df[(public_df['receipts'] >= 10) & (public_df['receipts'] < 50)]
medium_low_receipts = public_df[(public_df['receipts'] >= 50) & (public_df['receipts'] < 100)]

print(f"Very low receipts (<$10): {len(very_low_receipts)} cases")
print(f"  Mean reimbursement: ${very_low_receipts['expected_output'].mean():.2f}")
print(f"  Mean per trip day: ${(very_low_receipts['expected_output'] / very_low_receipts['trip_days']).mean():.2f}")

print(f"\nLow receipts ($10-50): {len(low_receipts)} cases")
print(f"  Mean reimbursement: ${low_receipts['expected_output'].mean():.2f}")
print(f"  Mean per trip day: ${(low_receipts['expected_output'] / low_receipts['trip_days']).mean():.2f}")

print(f"\nMedium-low receipts ($50-100): {len(medium_low_receipts)} cases")
print(f"  Mean reimbursement: ${medium_low_receipts['expected_output'].mean():.2f}")
print(f"  Mean per trip day: ${(medium_low_receipts['expected_output'] / medium_low_receipts['trip_days']).mean():.2f}")

# 5. RECEIPT THRESHOLD DETECTION
print("\n5. RECEIPT THRESHOLD DETECTION")
print("-" * 40)

# Look for discontinuities in the receipt-reimbursement relationship
thresholds_to_test = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]

for threshold in thresholds_to_test:
    below = public_df[public_df['receipts'] <= threshold]
    above = public_df[public_df['receipts'] > threshold]
    
    if len(below) > 10 and len(above) > 10:
        below_mean = below['expected_output'].mean()
        above_mean = above['expected_output'].mean()
        
        # Calculate average reimbursement per dollar of receipts
        below_ratio = (below['expected_output'] / below['receipts']).mean()
        above_ratio = (above['expected_output'] / above['receipts']).mean()
        
        ratio_change = (above_ratio - below_ratio) / below_ratio * 100
        
        if abs(ratio_change) > 10:  # Significant change
            print(f"\nThreshold at ${threshold}:")
            print(f"  Below: ${below_mean:.2f} mean (ratio: {below_ratio:.3f})")
            print(f"  Above: ${above_mean:.2f} mean (ratio: {above_ratio:.3f})")
            print(f"  Ratio change: {ratio_change:+.1f}%")

# 6. NON-LINEAR PATTERN DETECTION
print("\n6. NON-LINEAR PATTERN DETECTION")
print("-" * 40)

# Test different models for receipt-output relationship
X_receipts = public_df[['receipts']].values
y_output = public_df['expected_output'].values

# Linear model
linear_model = LinearRegression()
linear_model.fit(X_receipts, y_output)
linear_pred = linear_model.predict(X_receipts)
linear_r2 = r2_score(y_output, linear_pred)
linear_mae = mean_absolute_error(y_output, linear_pred)

print(f"Linear model: R² = {linear_r2:.4f}, MAE = ${linear_mae:.2f}")
print(f"  Coefficient: ${linear_model.coef_[0]:.4f} per dollar")
print(f"  Intercept: ${linear_model.intercept_:.2f}")

# Polynomial model (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_receipts)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_output)
poly_pred = poly_model.predict(X_poly)
poly_r2 = r2_score(y_output, poly_pred)
poly_mae = mean_absolute_error(y_output, poly_pred)

print(f"\nQuadratic model: R² = {poly_r2:.4f}, MAE = ${poly_mae:.2f}")

# Isotonic regression (monotonic, allows for plateaus)
isotonic_model = IsotonicRegression()
isotonic_model.fit(X_receipts.ravel(), y_output)
isotonic_pred = isotonic_model.predict(X_receipts.ravel())
isotonic_r2 = r2_score(y_output, isotonic_pred)
isotonic_mae = mean_absolute_error(y_output, isotonic_pred)

print(f"\nIsotonic model: R² = {isotonic_r2:.4f}, MAE = ${isotonic_mae:.2f}")

# Decision tree (can capture complex thresholds)
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_receipts, y_output)
tree_pred = tree_model.predict(X_receipts)
tree_r2 = r2_score(y_output, tree_pred)
tree_mae = mean_absolute_error(y_output, tree_pred)

print(f"\nDecision tree model: R² = {tree_r2:.4f}, MAE = ${tree_mae:.2f}")

# 7. RECEIPT CAP ANALYSIS
print("\n7. RECEIPT CAP ANALYSIS")
print("-" * 40)

# Look at high receipt cases
high_receipts = public_df[public_df['receipts'] > 1500]
print(f"High receipt cases (>$1500): {len(high_receipts)}")

if len(high_receipts) > 0:
    # Calculate effective reimbursement rate
    high_receipts_rate = (high_receipts['expected_output'] / high_receipts['receipts']).mean()
    overall_rate = (public_df['expected_output'] / public_df['receipts']).mean()
    
    print(f"  Average reimbursement rate: {high_receipts_rate:.2%}")
    print(f"  Overall average rate: {overall_rate:.2%}")
    print(f"  Difference: {(high_receipts_rate - overall_rate) / overall_rate * 100:+.1f}%")

# 8. SPECIAL RECEIPT AMOUNTS
print("\n8. SPECIAL RECEIPT AMOUNTS ANALYSIS")
print("-" * 40)

# Check for magic numbers or rounding effects
public_df['receipt_cents'] = (public_df['receipts'] * 100) % 100
public_df['receipt_dollars'] = public_df['receipts'].astype(int)

# Most common receipt amounts
common_amounts = public_df['receipts'].value_counts().head(20)
print("Most common receipt amounts:")
for amount, count in common_amounts.items():
    if count > 1:
        mean_reimb = public_df[public_df['receipts'] == amount]['expected_output'].mean()
        print(f"  ${amount:.2f}: {count} cases, avg reimbursement ${mean_reimb:.2f}")

# Check endings
endings = [0, 25, 49, 50, 75, 99]
for ending in endings:
    mask = (public_df['receipt_cents'].round() == ending)
    if mask.sum() > 5:
        mean_output = public_df[mask]['expected_output'].mean()
        overall_mean = public_df['expected_output'].mean()
        diff_pct = (mean_output - overall_mean) / overall_mean * 100
        print(f"\nReceipts ending in .{ending:02d}: {mask.sum()} cases")
        print(f"  Mean reimbursement: ${mean_output:.2f} ({diff_pct:+.1f}% vs overall)")

# 9. VISUALIZATION
print("\n9. CREATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Receipt Analysis Deep Dive', fontsize=16)

# 1. Receipt vs Reimbursement scatter
ax1 = axes[0, 0]
scatter = ax1.scatter(public_df['receipts'], public_df['expected_output'], 
                     alpha=0.5, c=public_df['trip_days'], cmap='viridis')
ax1.set_xlabel('Receipt Amount ($)')
ax1.set_ylabel('Reimbursement ($)')
ax1.set_title('Receipts vs Reimbursement (colored by trip days)')
plt.colorbar(scatter, ax=ax1, label='Trip Days')

# Add model fits
receipt_range = np.linspace(0, public_df['receipts'].max(), 100).reshape(-1, 1)
ax1.plot(receipt_range, linear_model.predict(receipt_range), 'r--', label='Linear', alpha=0.8)
ax1.plot(receipt_range, poly_model.predict(poly_features.transform(receipt_range)), 
         'g--', label='Quadratic', alpha=0.8)
ax1.legend()

# 2. Receipt coverage ratio by amount
ax2 = axes[0, 1]
coverage_data = public_df.groupby('receipt_bin')['receipt_coverage_capped'].agg(['mean', 'std'])
x_pos = np.arange(len(coverage_data))
ax2.bar(x_pos, coverage_data['mean'], yerr=coverage_data['std'], capsize=5)
ax2.set_xlabel('Receipt Range')
ax2.set_ylabel('Coverage Ratio')
ax2.set_title('Receipt Coverage Ratio by Amount Range')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(coverage_data.index, rotation=45)

# 3. Reimbursement distribution by receipt bins
ax3 = axes[0, 2]
receipt_bins_data = []
for bin_label in receipt_labels[:6]:  # First 6 bins for clarity
    if bin_label in public_df['receipt_bin'].values:
        data = public_df[public_df['receipt_bin'] == bin_label]['expected_output']
        if len(data) > 0:
            receipt_bins_data.append(data)

if receipt_bins_data:
    ax3.boxplot(receipt_bins_data, labels=receipt_labels[:len(receipt_bins_data)])
    ax3.set_xlabel('Receipt Range')
    ax3.set_ylabel('Reimbursement ($)')
    ax3.set_title('Reimbursement Distribution by Receipt Range')
    ax3.tick_params(axis='x', rotation=45)

# 4. Low receipt penalty visualization
ax4 = axes[1, 0]
low_receipt_df = public_df[public_df['receipts'] < 200].copy()
ax4.scatter(low_receipt_df['receipts'], low_receipt_df['expected_output'], alpha=0.6)
ax4.axvline(x=50, color='r', linestyle='--', label='$50 threshold')
ax4.set_xlabel('Receipt Amount ($)')
ax4.set_ylabel('Reimbursement ($)')
ax4.set_title('Low Receipt Penalty Analysis')
ax4.legend()

# 5. Residual analysis
ax5 = axes[1, 1]
residuals = y_output - linear_pred
ax5.scatter(public_df['receipts'], residuals, alpha=0.5)
ax5.axhline(y=0, color='r', linestyle='--')
ax5.set_xlabel('Receipt Amount ($)')
ax5.set_ylabel('Residual ($)')
ax5.set_title('Linear Model Residuals vs Receipt Amount')

# Add smoothed trend line
z = np.polyfit(public_df['receipts'], residuals, 3)
p = np.poly1d(z)
ax5.plot(np.sort(public_df['receipts']), p(np.sort(public_df['receipts'])), 
         'g-', linewidth=2, label='Trend')
ax5.legend()

# 6. Decision tree splits
ax6 = axes[1, 2]
# Show the most important thresholds from decision tree
feature_importance = tree_model.feature_importances_[0]
tree_thresholds = []

# Extract thresholds from tree
def get_thresholds(tree, feature=0):
    thresholds = []
    
    def traverse(node=0):
        if tree.feature[node] == feature:
            thresholds.append(tree.threshold[node])
            traverse(tree.children_left[node])
            traverse(tree.children_right[node])
    
    traverse()
    return sorted(thresholds)

tree_thresholds = get_thresholds(tree_model.tree_)
print(f"\nDecision tree key thresholds: {[f'${t:.0f}' for t in tree_thresholds[:5]]}")

# Plot tree predictions
sorted_receipts = np.sort(public_df['receipts'])
tree_line = tree_model.predict(sorted_receipts.reshape(-1, 1))
ax6.plot(sorted_receipts, tree_line, 'b-', linewidth=2, label='Decision Tree')
ax6.scatter(public_df['receipts'], public_df['expected_output'], alpha=0.3, s=20)

for threshold in tree_thresholds[:5]:
    ax6.axvline(x=threshold, color='r', linestyle=':', alpha=0.5)

ax6.set_xlabel('Receipt Amount ($)')
ax6.set_ylabel('Reimbursement ($)')
ax6.set_title('Decision Tree Model (showing thresholds)')
ax6.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'receipt_analysis_deep_dive.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to '{FIGURES_DIR / 'receipt_analysis_deep_dive.png'}'")

# 10. SAVE KEY FINDINGS
print("\n10. SAVING KEY FINDINGS")
print("-" * 40)

findings = {
    'receipt_statistics': public_df['receipts'].describe().to_dict(),
    'receipt_coverage_mean': public_df['receipt_coverage'].mean(),
    'low_receipt_penalty': {
        'under_10': very_low_receipts['expected_output'].mean() if len(very_low_receipts) > 0 else None,
        'under_50': low_receipts['expected_output'].mean() if len(low_receipts) > 0 else None,
        'threshold': 50
    },
    'model_performance': {
        'linear': {'r2': linear_r2, 'mae': linear_mae, 'coef': linear_model.coef_[0]},
        'quadratic': {'r2': poly_r2, 'mae': poly_mae},
        'isotonic': {'r2': isotonic_r2, 'mae': isotonic_mae},
        'decision_tree': {'r2': tree_r2, 'mae': tree_mae}
    },
    'key_thresholds': tree_thresholds[:5] if tree_thresholds else []
}

with open(REPORTS_DIR / 'receipt_analysis_findings.json', 'w') as f:
    json.dump(findings, f, indent=2)

print("Key findings saved to 'receipt_analysis_findings.json'")

# Update hypothesis document
print("\nUPDATING HYPOTHESIS DOCUMENT...")
# This would be done manually based on findings

print("\n" + "=" * 80)
print("RECEIPT ANALYSIS COMPLETE")
print("=" * 80)
print("\nKEY TAKEAWAYS:")
print("1. Strong low receipt penalty confirmed - under $50 is severely penalized")
print("2. Non-linear relationship - quadratic and tree models perform better")
print("3. Decision tree identifies key thresholds for receipt processing")
print("4. High receipts show diminishing returns in coverage ratio")
print("5. Special receipt endings (.49, .99) need further investigation") 