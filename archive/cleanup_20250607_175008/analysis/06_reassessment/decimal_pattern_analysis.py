"""
Decimal Pattern Analysis - Understanding the cent values
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

print("=" * 80)
print("DECIMAL PATTERN ANALYSIS")
print("=" * 80)

# Load the public cases with expected outputs
df = pd.read_csv(DATA_DIR / 'predictions' / 'public_cases_predictions_v0.5.csv')

# Extract cent values
df['expected_cents'] = (df['expected_output'] * 100) % 100
df['expected_cents_int'] = df['expected_cents'].round().astype(int)

# Analyze cent distribution
print("\nTop 20 most common cent values:")
cent_counts = df['expected_cents_int'].value_counts().head(20)
for cents, count in cent_counts.items():
    print(f"  .{cents:02d}: {count:3d} cases ({count/len(df)*100:4.1f}%)")

# Look for patterns in the cents
print("\n" + "=" * 40)
print("MATHEMATICAL PATTERNS IN CENTS")
print("=" * 40)

# Group by trip characteristics and see if cents follow patterns
print("\nAnalyzing cents by trip days:")
for days in range(1, 13):
    day_data = df[df['trip_days'] == days]
    if len(day_data) >= 10:  # Only analyze if enough data
        common_cents = day_data['expected_cents_int'].value_counts().head(3)
        print(f"\n{days} days ({len(day_data)} cases):")
        for cents, count in common_cents.items():
            print(f"  .{cents:02d}: {count} cases")

# Check if cents are related to inputs
print("\n" + "=" * 40)
print("CENTS CORRELATION WITH INPUTS")
print("=" * 40)

# Create features for correlation
df['receipt_cents'] = (df['receipts'] * 100) % 100
df['receipt_cents_int'] = df['receipt_cents'].round().astype(int)

# Check if output cents are related to receipt cents
print("\nChecking if output cents match receipt cents:")
df['cents_match'] = df['expected_cents_int'] == df['receipt_cents_int']
print(f"  Exact matches: {df['cents_match'].sum()} ({df['cents_match'].mean()*100:.1f}%)")

# Check for mathematical relationships
print("\nChecking mathematical relationships:")

# Test if cents = (trip_days * X) % 100 for various X
for multiplier in [10, 12, 15, 20, 24, 25, 30, 50, 75]:
    df[f'test_{multiplier}'] = (df['trip_days'] * multiplier) % 100
    matches = (df[f'test_{multiplier}'] == df['expected_cents']).sum()
    if matches > 50:
        print(f"  days * {multiplier} % 100: {matches} matches")

# Test if cents come from receipt calculations
receipt_multipliers = [0.1, 0.12, 0.15, 0.2, 0.24, 0.25, 0.3, 0.383, 0.5, 0.71]
for mult in receipt_multipliers:
    df[f'receipt_test'] = (df['receipts'] * mult * 100) % 100
    matches = (abs(df['receipt_test'] - df['expected_cents']) < 1).sum()
    if matches > 50:
        print(f"  receipts * {mult}: {matches} close matches")

# Analyze specific examples
print("\n" + "=" * 40)
print("EXAMPLE ANALYSIS")
print("=" * 40)

# Look at cases with specific cent patterns
for target_cents in [12, 24, 72, 94]:  # Most common patterns
    print(f"\nCases ending in .{target_cents:02d}:")
    examples = df[df['expected_cents_int'] == target_cents].head(3)
    for idx, row in examples.iterrows():
        print(f"  {row['trip_days']:.0f} days, {row['miles']:.0f} miles, "
              f"${row['receipts']:.2f} → ${row['expected_output']:.2f}")
        
        # Try to reverse engineer
        # Check if it's related to our known coefficients
        linear_calc = 57.80 + 46.69*row['trip_days'] + 0.51*row['miles'] + 0.71*row['receipts']
        print(f"    Linear calc would give: ${linear_calc:.2f}")

# Check for patterns in whole dollar amounts
print("\n" + "=" * 40)
print("WHOLE DOLLAR ANALYSIS")
print("=" * 40)

df['expected_dollars'] = df['expected_output'].astype(int)
df['is_round_hundred'] = df['expected_dollars'] % 100 == 0
df['is_round_fifty'] = df['expected_dollars'] % 50 == 0

print(f"Round hundreds ($X00): {df['is_round_hundred'].sum()}")
print(f"Round fifties ($X50): {df['is_round_fifty'].sum()}")
print(f"Exact whole dollars: {(df['expected_cents'] == 0).sum()}")

# Save detailed analysis
cent_analysis = df.groupby('expected_cents_int').agg({
    'trip_days': ['mean', 'std'],
    'miles': ['mean', 'std'],
    'receipts': ['mean', 'std'],
    'expected_output': ['count', 'mean']
}).round(2)

cent_analysis.to_csv(REPORTS_DIR / 'cent_pattern_analysis.csv')
print(f"\nDetailed cent analysis saved to: {REPORTS_DIR / 'cent_pattern_analysis.csv'}") 