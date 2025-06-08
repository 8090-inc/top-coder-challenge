import pandas as pd
import numpy as np
from models.cluster_models_optimized_fixed import calculate_reimbursement_v3_fixed
from models.cluster_models_optimized import calculate_reimbursement_v3

# Load data
df = pd.read_csv('public_cases_expected_outputs.csv')
df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100

print("TESTING V5.1 IMPROVEMENTS (Fixed Receipt Penalties)")
print("=" * 60)

# Calculate predictions with both versions
print("\nCalculating predictions...")
df['v3_original'] = df.apply(
    lambda r: calculate_reimbursement_v3(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)
df['v3_fixed'] = df.apply(
    lambda r: calculate_reimbursement_v3_fixed(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)

# Calculate errors
df['error_original'] = np.abs(df['v3_original'] - df['expected_output'])
df['error_fixed'] = np.abs(df['v3_fixed'] - df['expected_output'])

# Overall statistics
print(f"\nOverall Rule Engine Performance:")
print(f"  Original MAE: ${df['error_original'].mean():.2f}")
print(f"  Fixed MAE: ${df['error_fixed'].mean():.2f}")
print(f"  Improvement: ${df['error_original'].mean() - df['error_fixed'].mean():.2f}")
print(f"  Improvement %: {(df['error_original'].mean() - df['error_fixed'].mean()) / df['error_original'].mean() * 100:.1f}%")

# Analysis by receipt ending
print("\n\nPerformance by Receipt Ending:")
print("-" * 60)

# .99 endings
ending_99 = df[df['receipt_cents'] == 99]
print(f"\n.99 endings (n={len(ending_99)}):")
print(f"  Original MAE: ${ending_99['error_original'].mean():.2f}")
print(f"  Fixed MAE: ${ending_99['error_fixed'].mean():.2f}")
print(f"  Improvement: ${ending_99['error_original'].mean() - ending_99['error_fixed'].mean():.2f}")

# .49 endings  
ending_49 = df[df['receipt_cents'] == 49]
print(f"\n.49 endings (n={len(ending_49)}):")
print(f"  Original MAE: ${ending_49['error_original'].mean():.2f}")
print(f"  Fixed MAE: ${ending_49['error_fixed'].mean():.2f}")
print(f"  Improvement: ${ending_49['error_original'].mean() - ending_49['error_fixed'].mean():.2f}")

# Other endings
other = df[~df['receipt_cents'].isin([49, 99])]
print(f"\nOther endings (n={len(other)}):")
print(f"  Original MAE: ${other['error_original'].mean():.2f}")
print(f"  Fixed MAE: ${other['error_fixed'].mean():.2f}")
print(f"  Change: ${other['error_original'].mean() - other['error_fixed'].mean():.2f}")

# Show specific improvements for .99 cases
print("\n\nDetailed .99 Cases:")
print("-" * 100)
print(f"{'Days':>5} {'Miles':>7} {'Receipts':>10} {'Expected':>10} {'Original':>10} {'Fixed':>10} {'Improvement':>12}")
print("-" * 100)

for _, row in ending_99.iterrows():
    improvement = row['error_original'] - row['error_fixed']
    print(f"{row['trip_days']:5.0f} {row['miles']:7.0f} ${row['receipts']:9.2f} "
          f"${row['expected_output']:9.2f} ${row['v3_original']:9.2f} ${row['v3_fixed']:9.2f} "
          f"${improvement:11.2f}")

# Estimate v5.1 performance
print("\n\n" + "=" * 60)
print("ESTIMATED V5.1 PERFORMANCE:")
print("=" * 60)

# v5 had MAE of $77.41
# The rule engine improvement should translate to ensemble improvement
# but with some dampening factor (ensemble already corrects some errors)

rule_improvement = df['error_original'].mean() - df['error_fixed'].mean()
# Conservative estimate: 70% of rule engine improvement transfers to ensemble
estimated_v5_1_mae = 77.41 - (rule_improvement * 0.7)

print(f"\nCurrent v5 MAE: $77.41")
print(f"Rule engine improvement: ${rule_improvement:.2f}")
print(f"Estimated v5.1 MAE: ${estimated_v5_1_mae:.2f}")
print(f"Estimated improvement: ${77.41 - estimated_v5_1_mae:.2f} ({(77.41 - estimated_v5_1_mae)/77.41*100:.1f}%)")

if estimated_v5_1_mae < 75:
    print("\n✨ Significant improvement expected! Worth implementing v5.1")
else:
    print("\nModerate improvement expected.") 