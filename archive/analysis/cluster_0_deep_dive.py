import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from models.cluster_models_optimized import calculate_reimbursement_v3, calculate_cluster_0_optimized
from models.cluster_router import assign_cluster_v2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../public_cases_expected_outputs.csv')

# Get cluster 0 cases
df['cluster'] = df.apply(lambda r: assign_cluster_v2(r['trip_days'], r['miles'], r['receipts']), axis=1)
cluster_0 = df[df['cluster'] == '0'].copy()

print("CLUSTER 0 DEEP DIVE ANALYSIS")
print("=" * 60)
print(f"Total Cluster 0 cases: {len(cluster_0)} ({len(cluster_0)/len(df)*100:.1f}% of all cases)")

# Calculate predictions and errors
cluster_0['predicted'] = cluster_0.apply(
    lambda r: calculate_cluster_0_optimized(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)

# Apply receipt penalties
cluster_0['receipt_cents'] = (cluster_0['receipts'] * 100).astype(int) % 100
cluster_0['penalty_factor'] = 1.0
cluster_0.loc[cluster_0['receipt_cents'] == 49, 'penalty_factor'] = 0.341
cluster_0.loc[cluster_0['receipt_cents'] == 99, 'penalty_factor'] = 0.51
cluster_0['predicted_with_penalty'] = cluster_0['predicted'] * cluster_0['penalty_factor']

cluster_0['error'] = cluster_0['predicted_with_penalty'] - cluster_0['expected_output']
cluster_0['error_abs'] = np.abs(cluster_0['error'])

print(f"\nCurrent Performance:")
print(f"  MAE: ${cluster_0['error_abs'].mean():.2f}")
print(f"  Bias: ${cluster_0['error'].mean():.2f}")
print(f"  RMSE: ${np.sqrt((cluster_0['error']**2).mean()):.2f}")

# Analyze current formula
print(f"\n\nCurrent Cluster 0 Formula:")
print("  Standard: 182.45 + 52.57*days + 0.434*miles + 0.482*receipts")
print("  Special case: 9 days, 390-410 miles, 340-360 receipts → $913.29 (before penalty)")
print("  Receipt cap: $1800")

# Fit new linear model to see optimal coefficients
print("\n\nOptimal Linear Model (without penalties):")
X = cluster_0[['trip_days', 'miles', 'receipts']]
y = cluster_0['expected_output'] / cluster_0['penalty_factor']  # Remove penalty effect

model = LinearRegression()
model.fit(X, y)

print(f"  Intercept: {model.intercept_:.2f}")
print(f"  Days coefficient: {model.coef_[0]:.2f} (current: 52.57)")
print(f"  Miles coefficient: {model.coef_[1]:.3f} (current: 0.434)")
print(f"  Receipts coefficient: {model.coef_[2]:.3f} (current: 0.482)")

# Test new model
cluster_0['new_predicted'] = model.predict(X)
cluster_0['new_predicted_with_penalty'] = cluster_0['new_predicted'] * cluster_0['penalty_factor']
cluster_0['new_error'] = cluster_0['new_predicted_with_penalty'] - cluster_0['expected_output']
cluster_0['new_error_abs'] = np.abs(cluster_0['new_error'])

print(f"\nNew Model Performance:")
print(f"  MAE: ${cluster_0['new_error_abs'].mean():.2f}")
print(f"  Bias: ${cluster_0['new_error'].mean():.2f}")
print(f"  Improvement: ${cluster_0['error_abs'].mean() - cluster_0['new_error_abs'].mean():.2f}")

# Analyze problem cases
print("\n\nPROBLEM PATTERNS:")
print("-" * 60)

# 1. High error cases
high_error = cluster_0[cluster_0['error_abs'] > 200]
print(f"\n1. High Error Cases (>{200}, n={len(high_error)}):")
if len(high_error) > 0:
    print(f"   Average characteristics:")
    print(f"     Days: {high_error['trip_days'].mean():.1f}")
    print(f"     Miles: {high_error['miles'].mean():.0f}")
    print(f"     Receipts: ${high_error['receipts'].mean():.0f}")
    print(f"   Common patterns:")
    # Group by days
    days_dist = high_error['trip_days'].value_counts().head(3)
    for days, count in days_dist.items():
        print(f"     {days} days: {count} cases")

# 2. Receipt cap analysis
high_receipt = cluster_0[cluster_0['receipts'] > 1800]
print(f"\n2. High Receipt Cases (>${1800}, n={len(high_receipt)}):")
if len(high_receipt) > 0:
    print(f"   Current formula caps at $1800")
    print(f"   Their MAE: ${high_receipt['error_abs'].mean():.2f}")
    print(f"   Mean bias: ${high_receipt['error'].mean():.2f}")
    
    # Test different cap values
    print("\n   Testing different receipt caps:")
    for cap in [1500, 1800, 2000, 2500]:
        capped_receipts = np.minimum(high_receipt['receipts'], cap)
        test_pred = (182.45 + 52.57 * high_receipt['trip_days'] + 
                    0.434 * high_receipt['miles'] + 0.482 * capped_receipts)
        test_pred_with_penalty = test_pred * high_receipt['penalty_factor']
        test_error = np.abs(test_pred_with_penalty - high_receipt['expected_output']).mean()
        print(f"     Cap at ${cap}: MAE ${test_error:.2f}")

# 3. Penalty cases
penalty_cases = cluster_0[cluster_0['receipt_cents'].isin([49, 99])]
print(f"\n3. Receipt Penalty Cases (.49/.99, n={len(penalty_cases)}):")
if len(penalty_cases) > 0:
    print(f"   Their MAE: ${penalty_cases['error_abs'].mean():.2f}")
    print(f"   vs non-penalty: ${cluster_0[~cluster_0.index.isin(penalty_cases.index)]['error_abs'].mean():.2f}")

# 4. Efficiency analysis
cluster_0['miles_per_day'] = cluster_0['miles'] / cluster_0['trip_days']
cluster_0['receipts_per_day'] = cluster_0['receipts'] / cluster_0['trip_days']

print(f"\n4. Efficiency Patterns:")
# Low efficiency
low_eff = cluster_0[cluster_0['miles_per_day'] < 50]
print(f"   Low efficiency (<50 mi/day, n={len(low_eff)}): MAE ${low_eff['error_abs'].mean():.2f}")

# High efficiency  
high_eff = cluster_0[cluster_0['miles_per_day'] > 150]
print(f"   High efficiency (>150 mi/day, n={len(high_eff)}): MAE ${high_eff['error_abs'].mean():.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Cluster 0 Error Analysis', fontsize=14)

# Error vs days
ax1 = axes[0, 0]
ax1.scatter(cluster_0['trip_days'], cluster_0['error'], alpha=0.5)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel('Trip Days')
ax1.set_ylabel('Prediction Error ($)')
ax1.set_title('Error vs Trip Days')
ax1.grid(True, alpha=0.3)

# Error vs miles
ax2 = axes[0, 1]
ax2.scatter(cluster_0['miles'], cluster_0['error'], alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Miles')
ax2.set_ylabel('Prediction Error ($)')
ax2.set_title('Error vs Miles')
ax2.grid(True, alpha=0.3)

# Error vs receipts
ax3 = axes[1, 0]
ax3.scatter(cluster_0['receipts'], cluster_0['error'], alpha=0.5)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.axvline(x=1800, color='g', linestyle=':', label='Current cap')
ax3.set_xlabel('Receipts ($)')
ax3.set_ylabel('Prediction Error ($)')
ax3.set_title('Error vs Receipts')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Error distribution
ax4 = axes[1, 1]
ax4.hist(cluster_0['error'], bins=50, alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--')
ax4.axvline(x=cluster_0['error'].mean(), color='g', linestyle='-', label=f'Mean: ${cluster_0["error"].mean():.0f}')
ax4.set_xlabel('Prediction Error ($)')
ax4.set_ylabel('Count')
ax4.set_title('Error Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_0_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to cluster_0_analysis.png")

# Recommendations
print("\n\nRECOMMENDATIONS FOR CLUSTER 0:")
print("=" * 60)

improvements = []

# 1. Coefficient adjustment
coef_improvement = cluster_0['error_abs'].mean() - cluster_0['new_error_abs'].mean()
if coef_improvement > 5:
    improvements.append(("Adjust linear coefficients", coef_improvement))
    print(f"\n1. Update coefficients:")
    print(f"   Days: 52.57 → {model.coef_[0]:.2f}")
    print(f"   Miles: 0.434 → {model.coef_[1]:.3f}")
    print(f"   Receipts: 0.482 → {model.coef_[2]:.3f}")
    print(f"   Expected improvement: ${coef_improvement:.2f}")

# 2. Receipt cap
if len(high_receipt) > 10:
    print(f"\n2. Adjust receipt cap from $1800")
    print(f"   Consider dynamic cap or remove entirely")

# 3. Add sub-clusters
print(f"\n3. Consider sub-clustering Cluster 0:")
print(f"   - Low efficiency trips (<50 mi/day)")
print(f"   - High receipt trips (>$2000)")
print(f"   - Long trips (10+ days)")

# 4. Special cases
print(f"\n4. Add more special cases like the 9-day pattern")

total_improvement = sum(imp[1] for imp in improvements)
print(f"\n\nTotal potential improvement for Cluster 0: ~${total_improvement:.2f}")
print(f"Impact on overall MAE: ~${total_improvement * len(cluster_0) / 1000:.2f}")

plt.show() 