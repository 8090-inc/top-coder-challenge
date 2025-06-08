#!/usr/bin/env python3
"""
Investigate the mathematical formula behind cents patterns
"""

import pandas as pd
import numpy as np
from collections import Counter

# Load data
df = pd.read_csv('public_cases_predictions_v4.csv')

# Extract cents
df['expected_cents'] = (df['expected_output'] * 100).astype(int) % 100
df['expected_dollars'] = df['expected_output'].astype(int)

print("CENTS FORMULA INVESTIGATION")
print("="*60)

# Group by cents to look for patterns
cents_groups = df.groupby('expected_cents').agg({
    'trip_days': ['mean', 'std', 'count'],
    'miles': 'mean',
    'receipts': 'mean',
    'expected_output': ['mean', 'min', 'max']
}).round(2)

print("\nTop 20 most common cents values with their characteristics:")
top_cents = df['expected_cents'].value_counts().head(20)
for cents, count in top_cents.items():
    group = df[df['expected_cents'] == cents]
    print(f"\n.{cents:02d} ({count} cases):")
    print(f"  Avg output: ${group['expected_output'].mean():.2f}")
    print(f"  Avg days: {group['trip_days'].mean():.1f}")
    print(f"  Unique dollar amounts: {group['expected_dollars'].nunique()}")
    
    # Check for patterns in the dollar amounts
    dollar_counts = group['expected_dollars'].value_counts()
    if len(dollar_counts) <= 5:
        print(f"  Dollar amounts: {list(dollar_counts.index)}")

# Look for mathematical relationships
print("\n" + "="*60)
print("MATHEMATICAL PATTERNS")
print("="*60)

# Check if cents relate to inputs
print("\nChecking if cents relate to input features...")

# Simple modulo checks
df['days_mod_100'] = df['trip_days'] % 100
df['miles_mod_100'] = (df['miles'] * 100).astype(int) % 100
df['receipts_cents_input'] = (df['receipts'] * 100).astype(int) % 100

# Check correlations
for feature in ['days_mod_100', 'miles_mod_100', 'receipts_cents_input']:
    # Group by this feature and see if certain cents are more common
    cross_tab = pd.crosstab(df[feature], df['expected_cents'])
    if cross_tab.shape[0] < 20:  # Only if reasonable size
        print(f"\n{feature} vs expected_cents correlation:")
        # Find strong patterns
        for input_val in cross_tab.index[:10]:
            row = cross_tab.loc[input_val]
            if row.sum() > 5:  # At least 5 cases
                top_cents = row.nlargest(3)
                if top_cents.iloc[0] > row.sum() * 0.3:  # At least 30% concentration
                    print(f"  {feature}={input_val} -> cents {top_cents.index[0]} ({top_cents.iloc[0]}/{row.sum()} cases)")

# Check if cents follow a sequence based on case order
print("\n" + "="*60)
print("SEQUENCE ANALYSIS")
print("="*60)

# Sort by expected output to see if there's a pattern
df_sorted = df.sort_values('expected_output').reset_index(drop=True)
print("\nFirst 20 cases by expected output:")
print(df_sorted[['expected_output', 'expected_cents', 'trip_days', 'miles', 'receipts']].head(20).to_string(index=False))

# Check for arithmetic sequences in cents
cents_sequence = df_sorted['expected_cents'].values[:50]
print("\nFirst 50 cents in order of expected output:")
print(' '.join([f"{c:02d}" for c in cents_sequence]))

# Analyze specific common cents values
print("\n" + "="*60)
print("DEEP DIVE: COMMON CENTS PATTERNS")
print("="*60)

common_cents = [24, 12, 94, 72, 16, 68, 34]
for cents in common_cents:
    cases = df[df['expected_cents'] == cents]
    if len(cases) > 10:
        print(f"\nCases ending in .{cents:02d}:")
        # Look for common factors
        print(f"  Count: {len(cases)}")
        print(f"  Receipt endings: {cases['receipts_cents_input'].value_counts().head(5).to_dict()}")
        print(f"  Days distribution: {cases['trip_days'].value_counts().head(5).to_dict()}")
        
        # Check if there's a formula
        sample = cases.head(3)
        for _, row in sample.iterrows():
            total_calc = row['trip_days'] * 100 + row['miles'] + row['receipts']
            print(f"  Example: {row['trip_days']}d + {row['miles']:.0f}mi + ${row['receipts']:.2f} = ${row['expected_output']:.2f}")
            print(f"    (days*100 + miles + receipts) % 100 = {int(total_calc) % 100}")

# Final check - is there a hidden calculation?
print("\n" + "="*60)
print("REVERSE ENGINEERING ATTEMPTS")
print("="*60)

# Try various formulas
formulas = [
    ('(days + miles + receipts) % 100', lambda r: int(r['trip_days'] + r['miles'] + r['receipts']) % 100),
    ('(days*10 + miles/10) % 100', lambda r: int(r['trip_days']*10 + r['miles']/10) % 100),
    ('(output/10) % 100', lambda r: int(r['expected_output']/10) % 100),
    ('(days*miles) % 100', lambda r: int(r['trip_days'] * r['miles']) % 100),
]

for name, formula in formulas:
    df['test_cents'] = df.apply(formula, axis=1)
    matches = (df['test_cents'] == df['expected_cents']).sum()
    if matches > 50:  # If more than 5% match
        print(f"\n{name}: {matches} matches ({matches/len(df)*100:.1f}%)")
        if matches > 100:
            # Show some examples
            matching = df[df['test_cents'] == df['expected_cents']].head(5)
            print("Examples where it works:")
            print(matching[['trip_days', 'miles', 'receipts', 'expected_output', 'expected_cents']].to_string(index=False)) 