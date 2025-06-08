#!/usr/bin/env python3
"""
Investigate the .49 receipt ending penalty issue
"""

import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('public_cases_predictions_v4.csv')

# Add receipt ending info
df['receipts_cents'] = (df['receipts'] * 100).astype(int) % 100
df['has_49_ending'] = df['receipts_cents'] == 49
df['has_99_ending'] = df['receipts_cents'] == 99

# Focus on .49 ending cases
cases_49 = df[df['has_49_ending']].copy()
print(f"Total cases with .49 ending: {len(cases_49)}")
print(f"Average error for .49 cases: ${cases_49['abs_error'].mean():.2f}")
print(f"Max error for .49 cases: ${cases_49['abs_error'].max():.2f}")

# Compare with all cases
print(f"\nComparison:")
print(f"Average error (all cases): ${df['abs_error'].mean():.2f}")
print(f"Average error (.49 cases): ${cases_49['abs_error'].mean():.2f}")

# Look at distribution of errors for .49 cases
print("\n.49 cases with highest errors:")
print("-" * 60)
top_49_errors = cases_49.nlargest(10, 'abs_error')[['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'abs_error', 'error']]
print(top_49_errors.to_string(index=False))

# Calculate what the prediction would be without penalty
# Our penalty is 0.341x, so divide by 0.341 to get pre-penalty amount
cases_49['predicted_no_penalty'] = cases_49['predicted'] / 0.341
cases_49['would_be_error'] = abs(cases_49['predicted_no_penalty'] - cases_49['expected_output'])

print("\nWhat if we didn't apply the .49 penalty?")
print("-" * 60)
comparison = cases_49[['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'predicted_no_penalty', 'abs_error', 'would_be_error']].head(10)
print(comparison.to_string(index=False))

# Check if removing penalty would improve overall
better_without = (cases_49['would_be_error'] < cases_49['abs_error']).sum()
worse_without = (cases_49['would_be_error'] > cases_49['abs_error']).sum()
print(f"\nCases that would be better without penalty: {better_without}")
print(f"Cases that would be worse without penalty: {worse_without}")

# Specific case: 7 days, 194 miles, $202.49
specific_case = cases_49[(cases_49['trip_days'] == 7) & (cases_49['miles'] == 194)]
if len(specific_case) > 0:
    row = specific_case.iloc[0]
    print(f"\nSpecific case analysis (7d, 194mi, $202.49):")
    print(f"Expected: ${row['expected_output']:.2f}")
    print(f"Current prediction: ${row['predicted']:.2f} (error: ${row['abs_error']:.2f})")
    print(f"Without .49 penalty: ${row['predicted_no_penalty']:.2f} (error: ${row['would_be_error']:.2f})")
    
    # Calculate what multiplier would give correct result
    ideal_multiplier = row['expected_output'] / row['predicted_no_penalty']
    print(f"Ideal multiplier for this case: {ideal_multiplier:.3f} (vs current 0.341)")

# Look for patterns in .49 cases that need different treatment
print("\n\nPattern analysis for .49 cases:")
print("-" * 60)

# Group by trip days
print("\nBy trip duration:")
day_groups = cases_49.groupby('trip_days').agg({
    'abs_error': ['count', 'mean', 'max'],
    'would_be_error': 'mean'
}).round(2)
print(day_groups)

# Check if low miles/receipts .49 cases are different
cases_49['miles_per_day'] = cases_49['miles'] / cases_49['trip_days']
cases_49['receipts_per_day'] = cases_49['receipts'] / cases_49['trip_days']

low_efficiency_49 = cases_49[cases_49['miles_per_day'] < 50]
print(f"\n\nLow miles/day (<50) .49 cases: {len(low_efficiency_49)}")
if len(low_efficiency_49) > 0:
    print("These cases:")
    print(low_efficiency_49[['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'abs_error']].to_string(index=False)) 