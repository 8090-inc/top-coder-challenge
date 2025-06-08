import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from models.cluster_models_optimized import calculate_reimbursement_v3
from models.cluster_router import assign_cluster_v2

# Load data
df = pd.read_csv('../public_cases_expected_outputs.csv')

# Calculate rule engine predictions and errors
print("RULE ENGINE IMPROVEMENT ANALYSIS")
print("=" * 60)

df['rule_pred'] = df.apply(
    lambda r: calculate_reimbursement_v3(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)
df['rule_error'] = df['rule_pred'] - df['expected_output']
df['rule_error_abs'] = np.abs(df['rule_error'])
df['cluster'] = df.apply(
    lambda r: assign_cluster_v2(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)

print(f"\nCurrent Rule Engine Performance:")
print(f"  MAE: ${df['rule_error_abs'].mean():.2f}")
print(f"  RMSE: ${np.sqrt((df['rule_error']**2).mean()):.2f}")
print(f"  Mean bias: ${df['rule_error'].mean():.2f} (negative = underprediction)")
print(f"  Cases with error > $200: {(df['rule_error_abs'] > 200).sum()}")
print(f"  Cases with error > $100: {(df['rule_error_abs'] > 100).sum()}")

# Analyze systematic biases by cluster
print("\n\nSystematic Biases by Cluster:")
print("-" * 60)
cluster_stats = df.groupby('cluster').agg({
    'rule_error': ['mean', 'std', 'count'],
    'rule_error_abs': 'mean'
}).round(2)

print(cluster_stats)

# Find patterns in high-error cases
high_error = df[df['rule_error_abs'] > 200].copy()
print(f"\n\nHigh Error Cases (>{200}, n={len(high_error)}):")
print("-" * 60)

# Analyze characteristics
print("\nCharacteristics of high-error cases:")
print(f"  Average days: {high_error['trip_days'].mean():.1f} (vs {df['trip_days'].mean():.1f} overall)")
print(f"  Average miles: {high_error['miles'].mean():.0f} (vs {df['miles'].mean():.0f} overall)")
print(f"  Average receipts: ${high_error['receipts'].mean():.0f} (vs ${df['receipts'].mean():.0f} overall)")

# Check for receipt patterns
high_error['receipt_cents'] = (high_error['receipts'] * 100).astype(int) % 100
print(f"\n  Receipt endings in high-error cases:")
print(f"    .49 endings: {(high_error['receipt_cents'] == 49).sum()}")
print(f"    .99 endings: {(high_error['receipt_cents'] == 99).sum()}")
print(f"    Other: {(~high_error['receipt_cents'].isin([49, 99])).sum()}")

# Analyze specific improvement opportunities
print("\n\nSPECIFIC IMPROVEMENT OPPORTUNITIES:")
print("=" * 60)

# 1. Receipt penalties
penalty_cases = df[df['receipts'].apply(lambda x: int(x * 100) % 100).isin([49, 99])]
penalty_error = penalty_cases['rule_error_abs'].mean()
other_error = df[~df.index.isin(penalty_cases.index)]['rule_error_abs'].mean()
print(f"\n1. Fix Receipt Penalties:")
print(f"   Current MAE for .49/.99 cases: ${penalty_error:.2f}")
print(f"   Current MAE for other cases: ${other_error:.2f}")
print(f"   Potential improvement: ~${(penalty_error - other_error) * len(penalty_cases) / len(df):.2f}")

# 2. Cluster reassignment
print(f"\n2. Cluster Reassignment Opportunities:")
# Find cases where neighboring clusters might be better
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    if len(cluster_data) > 10:  # Only analyze clusters with enough data
        # Look for outliers within cluster
        cluster_errors = cluster_data['rule_error_abs']
        outliers = cluster_data[cluster_errors > cluster_errors.quantile(0.9)]
        if len(outliers) > 0:
            print(f"\n   Cluster {cluster} outliers ({len(outliers)} cases):")
            print(f"     Mean error: ${outliers['rule_error_abs'].mean():.2f}")
            print(f"     Characteristics: {outliers['trip_days'].mean():.1f}d, "
                  f"{outliers['miles'].mean():.0f}mi, ${outliers['receipts'].mean():.0f}")

# 3. Linear coefficient refinement
print(f"\n3. Linear Coefficient Refinement:")
print("   Clusters with systematic bias (mean error > $50):")
biased_clusters = cluster_stats[cluster_stats[('rule_error', 'mean')].abs() > 50]
for cluster in biased_clusters.index:
    bias = cluster_stats.loc[cluster, ('rule_error', 'mean')]
    count = cluster_stats.loc[cluster, ('rule_error', 'count')]
    print(f"     Cluster {cluster}: {bias:+.0f} bias ({count} cases)")

# 4. New special cases
print(f"\n4. Potential New Special Cases:")
# Look for tight patterns with consistent errors
df['pattern'] = df['trip_days'].astype(str) + 'd_' + (df['miles'] // 100).astype(str) + '00mi'
pattern_errors = df.groupby('pattern').agg({
    'rule_error': ['mean', 'std', 'count'],
    'rule_error_abs': 'mean'
})
significant_patterns = pattern_errors[
    (pattern_errors[('rule_error', 'count')] >= 3) & 
    (pattern_errors[('rule_error_abs', 'mean')] > 150)
].sort_values(('rule_error_abs', 'mean'), ascending=False)

print("\n   Patterns with consistent high errors:")
for pattern in significant_patterns.head(5).index:
    mean_error = significant_patterns.loc[pattern, ('rule_error', 'mean')]
    count = significant_patterns.loc[pattern, ('rule_error', 'count')]
    mae = significant_patterns.loc[pattern, ('rule_error_abs', 'mean')]
    print(f"     {pattern}: MAE ${mae:.0f}, bias {mean_error:+.0f} ({count} cases)")

# 5. Receipt cap adjustments
print(f"\n5. Receipt Cap Analysis:")
high_receipt_cases = df[df['receipts'] > 1500]
print(f"   High receipt cases (>${1500}): {len(high_receipt_cases)}")
print(f"   Their MAE: ${high_receipt_cases['rule_error_abs'].mean():.2f}")
print(f"   Systematic bias: {high_receipt_cases['rule_error'].mean():+.2f}")

# Summary
print("\n\nSUMMARY OF IMPROVEMENT POTENTIAL:")
print("=" * 60)

improvements = {
    'Fix receipt penalties': (penalty_error - other_error) * len(penalty_cases) / len(df),
    'Reduce cluster 0 bias': abs(cluster_stats.loc['0', ('rule_error', 'mean')]) * 0.5 * cluster_stats.loc['0', ('rule_error', 'count')] / len(df),
    'Add special patterns': 10.0,  # Conservative estimate
    'Refine coefficients': 5.0,    # Conservative estimate
}

total_improvement = sum(improvements.values())

print("\nEstimated improvements:")
for improvement, value in improvements.items():
    print(f"  {improvement}: ~${value:.2f}")

print(f"\nTotal potential improvement: ~${total_improvement:.2f}")
print(f"Estimated new MAE: ~${df['rule_error_abs'].mean() - total_improvement:.2f}")
print(f"That would be a {total_improvement / df['rule_error_abs'].mean() * 100:.1f}% improvement")

# Compare to v5
v5_mae = 77.41
rule_mae = df['rule_error_abs'].mean()
print(f"\n\nContext:")
print(f"  Current rule engine MAE: ${rule_mae:.2f}")
print(f"  Current v5 (rule + ML) MAE: ${v5_mae:.2f}")
print(f"  ML correction value: ${rule_mae - v5_mae:.2f}")
print(f"\nEven with improvements, ML ensemble adds significant value!") 