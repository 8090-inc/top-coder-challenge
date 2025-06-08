import pandas as pd
import numpy as np
from models.cluster_models_optimized import calculate_reimbursement_v3

# Load the v5 predictions
df = pd.read_csv('public_cases_predictions_v5.csv')

# Focus on .99 endings
df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100
ending_99 = df[df['receipt_cents'] == 99]

print("TESTING RECEIPT PENALTY APPLICATION IN V5")
print("=" * 60)
print(f"\nAnalyzing {len(ending_99)} cases with .99 endings...")
print("-" * 60)

# For each .99 case, check if the rule engine is applying penalties
for idx, row in ending_99.head(10).iterrows():
    # Get rule engine prediction
    rule_pred = calculate_reimbursement_v3(
        row['trip_days'], 
        row['miles'], 
        row['receipts']
    )
    
    # Calculate what it would be without penalty
    # First get the cluster assignment
    from models.cluster_router import assign_cluster_v2
    cluster = assign_cluster_v2(row['trip_days'], row['miles'], row['receipts'])
    
    # Calculate base amount without penalty
    if cluster == '0':
        from models.cluster_models_optimized import calculate_cluster_0_optimized
        base_amount = calculate_cluster_0_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '1a':
        from models.cluster_models_optimized import calculate_cluster_1a_optimized
        base_amount = calculate_cluster_1a_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '1b':
        from models.cluster_models_optimized import calculate_cluster_1b_optimized
        base_amount = calculate_cluster_1b_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '2':
        from models.cluster_models_optimized import calculate_cluster_2_optimized
        base_amount = calculate_cluster_2_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '3':
        from models.cluster_models_optimized import calculate_cluster_3_optimized
        base_amount = calculate_cluster_3_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '4':
        from models.cluster_models_optimized import calculate_cluster_4_optimized
        base_amount = calculate_cluster_4_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '5':
        from models.cluster_models_optimized import calculate_cluster_5_optimized
        base_amount = calculate_cluster_5_optimized(row['trip_days'], row['miles'], row['receipts'])
    elif cluster == '6':
        from models.cluster_models_optimized import calculate_cluster_6_optimized
        base_amount = calculate_cluster_6_optimized(row['trip_days'], row['miles'], row['receipts'])
    else:
        base_amount = calculate_cluster_0_optimized(row['trip_days'], row['miles'], row['receipts'])
    
    # Expected penalty application
    expected_with_penalty = base_amount * 0.51
    
    print(f"\nCase {idx}: {row['trip_days']:.0f} days, {row['miles']:.0f} miles, ${row['receipts']:.2f}")
    print(f"  Cluster: {cluster}")
    print(f"  Base amount (no penalty): ${base_amount:.2f}")
    print(f"  Expected with penalty: ${expected_with_penalty:.2f}")
    print(f"  Actual rule engine: ${rule_pred:.2f}")
    print(f"  Penalty applied correctly: {'YES' if abs(rule_pred - expected_with_penalty) < 0.01 else 'NO'}")
    print(f"  v5 predicted: ${row['predicted']:.2f}")
    print(f"  Actual expected: ${row['expected_output']:.2f}")
    print(f"  v5 error: ${row['predicted'] - row['expected_output']:.2f}")

# Now check if v5 is overcorrecting for the penalty
print("\n\n" + "=" * 60)
print("ANALYSIS: Is v5 overcorrecting for penalties?")
print("=" * 60)

# Compare rule engine predictions with expected outputs for .99 cases
ending_99['rule_pred'] = ending_99.apply(
    lambda r: calculate_reimbursement_v3(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)
ending_99['rule_error'] = ending_99['rule_pred'] - ending_99['expected_output']
ending_99['v5_error'] = ending_99['predicted'] - ending_99['expected_output']

print(f"\n.99 ending cases (n={len(ending_99)}):")
print(f"  Average rule engine error: ${ending_99['rule_error'].mean():.2f}")
print(f"  Average v5 error: ${ending_99['v5_error'].mean():.2f}")
print(f"  Rule engine MAE: ${ending_99['rule_error'].abs().mean():.2f}")
print(f"  v5 MAE: ${ending_99['v5_error'].abs().mean():.2f}")

# Check if v5 is consistently under or over predicting
print(f"\n  Rule engine underpredicts: {(ending_99['rule_error'] < 0).sum()}/{len(ending_99)} cases")
print(f"  v5 underpredicts: {(ending_99['v5_error'] < 0).sum()}/{len(ending_99)} cases")

# Compare with non-penalty cases
other = df[~df['receipt_cents'].isin([49, 99])]
other['rule_pred'] = other.apply(
    lambda r: calculate_reimbursement_v3(r['trip_days'], r['miles'], r['receipts']), 
    axis=1
)
other['rule_error'] = other['rule_pred'] - other['expected_output']
other['v5_error'] = other['predicted'] - other['expected_output']

print(f"\nOther endings (n={len(other)}):")
print(f"  Average rule engine error: ${other['rule_error'].mean():.2f}")
print(f"  Average v5 error: ${other['v5_error'].mean():.2f}")
print(f"  Rule engine MAE: ${other['rule_error'].abs().mean():.2f}")
print(f"  v5 MAE: ${other['v5_error'].abs().mean():.2f}")

print(f"\n\nCONCLUSION:")
print("-" * 60)
improvement_potential = ending_99['v5_error'].abs().mean() - other['v5_error'].abs().mean()
print(f"Fixing .99 endings could reduce MAE by approximately: ${improvement_potential:.2f}") 