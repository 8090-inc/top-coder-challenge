import pandas as pd
import numpy as np
from models.error_correction_patterns import ErrorCorrectionPatterns

# Load v5 predictions with errors
df = pd.read_csv('public_cases_predictions_v5.csv')
df['error'] = np.abs(df['predicted'] - df['expected_output'])

print("TESTING V5 IMPROVEMENTS")
print("=" * 60)
print(f"Current v5 MAE: ${df['error'].mean():.2f}")
print(f"Current v5 MAPE: {(df['error'] / df['expected_output'] * 100).mean():.1f}%")
print(f"Max Error: ${df['error'].max():.2f}")

# Test Pattern Corrections
print("\n\nTesting Error Correction Patterns:")
print("-" * 60)

corrected_predictions = []
correction_counts = {}

for idx, row in df.iterrows():
    base_pred = row['predicted']
    corrected = ErrorCorrectionPatterns.apply_corrections(
        row['trip_days'], 
        row['miles'], 
        row['receipts'],
        base_pred
    )
    corrected_predictions.append(corrected)
    
    # Track which corrections were applied
    if corrected != base_pred:
        edge_cases = ErrorCorrectionPatterns.identify_edge_cases(
            row['trip_days'], row['miles'], row['receipts']
        )
        for case in edge_cases:
            correction_counts[case] = correction_counts.get(case, 0) + 1

df['corrected_predicted'] = corrected_predictions
df['corrected_error'] = np.abs(df['corrected_predicted'] - df['expected_output'])

# Calculate improvements
new_mae = df['corrected_error'].mean()
improvement = df['error'].mean() - new_mae
improvement_pct = improvement / df['error'].mean() * 100

print(f"\nWith Error Corrections:")
print(f"New MAE: ${new_mae:.2f}")
print(f"Improvement: ${improvement:.2f} ({improvement_pct:.1f}%)")
print(f"New MAPE: {(df['corrected_error'] / df['expected_output'] * 100).mean():.1f}%")

# Analyze specific pattern improvements
print("\n\nPattern-Specific Improvements:")

# Very high receipts
high_receipts = df[df['receipts'] > 2000]
if len(high_receipts) > 0:
    orig_error = high_receipts['error'].mean()
    new_error = high_receipts['corrected_error'].mean()
    print(f"\nVery high receipts (>${2000}):")
    print(f"  Original MAE: ${orig_error:.2f}")
    print(f"  Corrected MAE: ${new_error:.2f}")
    print(f"  Improvement: ${orig_error - new_error:.2f}")

# Single day trips
single_day = df[df['trip_days'] == 1]
if len(single_day) > 0:
    orig_error = single_day['error'].mean()
    new_error = single_day['corrected_error'].mean()
    print(f"\nSingle day trips:")
    print(f"  Original MAE: ${orig_error:.2f}")
    print(f"  Corrected MAE: ${new_error:.2f}")
    print(f"  Improvement: ${orig_error - new_error:.2f}")

# Long trips
long_trips = df[df['trip_days'] >= 10]
if len(long_trips) > 0:
    orig_error = long_trips['error'].mean()
    new_error = long_trips['corrected_error'].mean()
    print(f"\nLong trips (>=10 days):")
    print(f"  Original MAE: ${orig_error:.2f}")
    print(f"  Corrected MAE: ${new_error:.2f}")
    print(f"  Improvement: ${orig_error - new_error:.2f}")

# Edge case distribution
if correction_counts:
    print("\n\nEdge Cases Found:")
    for case, count in sorted(correction_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {case}: {count} cases")

# Find remaining high-error cases
remaining_high_error = df[df['corrected_error'] > 200]
print(f"\n\nRemaining high-error cases (>${200}): {len(remaining_high_error)}")
if len(remaining_high_error) > 0:
    print("\nTop 10 remaining errors:")
    print("-" * 100)
    print(f"{'Days':>5} {'Miles':>7} {'Receipts':>10} {'Expected':>10} {'Original':>10} {'Corrected':>10} {'Error':>8}")
    print("-" * 100)
    
    for idx in remaining_high_error.nlargest(10, 'corrected_error').index:
        row = df.loc[idx]
        print(f"{row['trip_days']:5.0f} {row['miles']:7.0f} ${row['receipts']:9.2f} "
              f"${row['expected_output']:9.2f} ${row['predicted']:9.2f} "
              f"${row['corrected_predicted']:9.2f} ${row['corrected_error']:7.2f}")

# Estimate combined improvement with cents prediction
print("\n\n" + "=" * 60)
print("ESTIMATED COMBINED IMPROVEMENTS:")
print("=" * 60)

# Assume cents classifier gets 80% accuracy (conservative)
cents_accuracy = 0.80
avg_cents_error = 50 * (1 - cents_accuracy) / 100  # Average cents error in dollars

combined_mae = new_mae - avg_cents_error * 0.5  # Conservative estimate
print(f"\nCurrent v5 MAE: ${df['error'].mean():.2f}")
print(f"With error corrections: ${new_mae:.2f}")
print(f"Estimated with cents prediction: ${combined_mae:.2f}")
print(f"Total potential improvement: ${df['error'].mean() - combined_mae:.2f} ({(df['error'].mean() - combined_mae) / df['error'].mean() * 100:.1f}%)")

if combined_mae < 70:
    print(f"\n✨ PROJECTED MAE: ${combined_mae:.2f} - Potential new record!")
else:
    print(f"\nProjected MAE: ${combined_mae:.2f} - Good improvement but more work needed") 