"""
Exact Match Analysis - Why are we getting 0 exact matches?
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
import calculate_reimbursement as calc

print("=" * 80)
print("EXACT MATCH ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_csv(DATA_DIR / 'predictions' / 'public_cases_predictions_v0.5.csv')

# Calculate exact match threshold
df['is_exact'] = abs(df['predicted'] - df['expected_output']) < 0.01
df['error'] = df['predicted'] - df['expected_output']
df['abs_error'] = abs(df['error'])

print(f"\nTotal cases: {len(df)}")
print(f"Exact matches: {df['is_exact'].sum()} ({df['is_exact'].mean()*100:.1f}%)")
print(f"Mean absolute error: ${df['abs_error'].mean():.2f}")

# Analyze error distribution
print("\n" + "=" * 40)
print("ERROR DISTRIBUTION")
print("=" * 40)

# Check if errors are systematic
print(f"\nMean error (bias): ${df['error'].mean():.2f}")
print(f"Std error: ${df['error'].std():.2f}")

# Error ranges
error_ranges = [
    (0, 1, "< $1"),
    (1, 10, "$1-10"),
    (10, 50, "$10-50"),
    (50, 100, "$50-100"),
    (100, 200, "$100-200"),
    (200, 500, "$200-500"),
    (500, float('inf'), "> $500")
]

print("\nError distribution:")
for low, high, label in error_ranges:
    count = ((df['abs_error'] >= low) & (df['abs_error'] < high)).sum()
    pct = count / len(df) * 100
    print(f"  {label:>10}: {count:4d} ({pct:5.1f}%)")

# Check for patterns in exact/near-exact cases
close_matches = df[df['abs_error'] < 1]
print(f"\nClose matches (<$1 error): {len(close_matches)}")
if len(close_matches) > 0:
    print("\nClose match examples:")
    for idx, row in close_matches.head(10).iterrows():
        print(f"  Days: {row['trip_days']:.0f}, Miles: {row['miles']:.0f}, "
              f"Receipts: ${row['receipts']:.2f}")
        print(f"    Expected: ${row['expected_output']:.2f}, "
              f"Predicted: ${row['predicted']:.2f}, "
              f"Error: ${row['abs_error']:.2f}")

# Check if outputs are rounded
print("\n" + "=" * 40)
print("OUTPUT ROUNDING ANALYSIS")
print("=" * 40)

# Check decimal places in expected outputs
df['expected_cents'] = (df['expected_output'] * 100) % 100
df['predicted_cents'] = (df['predicted'] * 100) % 100

print(f"\nExpected outputs with .00: {(df['expected_cents'] == 0).sum()}")
print(f"Expected outputs with other cents: {(df['expected_cents'] != 0).sum()}")

# Show distribution of cents
print("\nMost common cent values in expected outputs:")
cent_counts = df['expected_cents'].value_counts().head(10)
for cents, count in cent_counts.items():
    print(f"  .{cents:02.0f}: {count} cases")

# Check if our predictions are too "round"
print(f"\nOur predictions with .00: {(df['predicted_cents'] == 0).sum()}")
print(f"Our predictions with other cents: {(df['predicted_cents'] != 0).sum()}")

# Analyze specific patterns
print("\n" + "=" * 40)
print("PATTERN ANALYSIS")
print("=" * 40)

# Check receipt endings impact
df['receipt_cents'] = (df['receipts'] * 100) % 100
df['has_49'] = df['receipt_cents'] == 49
df['has_99'] = df['receipt_cents'] == 99

print(f"\nCases with .49 receipts: {df['has_49'].sum()}")
print(f"Cases with .99 receipts: {df['has_99'].sum()}")

# Check if penalty is being applied correctly
penalized = df[df['has_49'] | df['has_99']]
if len(penalized) > 0:
    print(f"\nPenalized cases analysis:")
    print(f"  Mean error: ${penalized['abs_error'].mean():.2f}")
    print(f"  vs non-penalized: ${df[~(df['has_49'] | df['has_99'])]['abs_error'].mean():.2f}")

# Test a few cases manually
print("\n" + "=" * 40)
print("MANUAL VERIFICATION")
print("=" * 40)

test_cases = [
    (5, 300, 750.50),
    (1, 500, 1000),
    (7, 1000, 1100),
    (10, 800, 1500),
    (3, 200, 500)
]

print("\nTesting sample calculations:")
for days, miles, receipts in test_cases:
    result = calc.calculate_reimbursement(days, miles, receipts, version='v0.5')
    print(f"  {days} days, {miles} miles, ${receipts} → ${result:.2f}")

# Save detailed error analysis
error_analysis = df[['trip_days', 'miles', 'receipts', 'expected_output', 
                     'predicted', 'error', 'abs_error']].copy()
error_analysis = error_analysis.sort_values('abs_error', ascending=False)
error_analysis.to_csv(REPORTS_DIR / 'v05_error_analysis.csv', index=False)
print(f"\nDetailed error analysis saved to: {REPORTS_DIR / 'v05_error_analysis.csv'}") 