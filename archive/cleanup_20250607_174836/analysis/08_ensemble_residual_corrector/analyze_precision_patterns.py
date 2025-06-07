#!/usr/bin/env python3
"""
Analyze precision patterns - why 0 exact matches?
"""

import pandas as pd
import numpy as np
from collections import Counter

# Load predictions
df = pd.read_csv('public_cases_predictions_v4.csv')

# Analyze error distribution
print("ERROR DISTRIBUTION ANALYSIS")
print("="*60)

# Basic stats
print(f"Total cases: {len(df)}")
print(f"MAE: ${df['abs_error'].mean():.2f}")
print(f"Median error: ${df['abs_error'].median():.2f}")
print(f"Std dev: ${df['abs_error'].std():.2f}")

# Error buckets
print("\nError buckets:")
buckets = [0.01, 1, 5, 10, 25, 50, 100, 200]
for i in range(len(buckets)-1):
    count = ((df['abs_error'] >= buckets[i]) & (df['abs_error'] < buckets[i+1])).sum()
    pct = count / len(df) * 100
    print(f"  ${buckets[i]:.2f} - ${buckets[i+1]:.2f}: {count} cases ({pct:.1f}%)")

# Analyze cents patterns
print("\n" + "="*60)
print("CENTS PATTERN ANALYSIS")
print("="*60)

# Expected cents
df['expected_cents'] = (df['expected_output'] * 100).astype(int) % 100
df['predicted_cents'] = (df['predicted'] * 100).astype(int) % 100

print("\nTop 10 expected cents patterns:")
expected_counts = df['expected_cents'].value_counts().head(10)
for cents, count in expected_counts.items():
    print(f"  .{cents:02d}: {count} cases ({count/len(df)*100:.1f}%)")

print("\nTop 10 predicted cents patterns:")
predicted_counts = df['predicted_cents'].value_counts().head(10)
for cents, count in predicted_counts.items():
    print(f"  .{cents:02d}: {count} cases ({count/len(df)*100:.1f}%)")

# Compare cents match rate
df['cents_match'] = df['expected_cents'] == df['predicted_cents']
print(f"\nCents exact match rate: {df['cents_match'].sum()}/{len(df)} ({df['cents_match'].mean()*100:.1f}%)")

# Analyze near misses
print("\n" + "="*60)
print("NEAR MISS ANALYSIS")
print("="*60)

# Cases within $0.01-$1.00
near_misses = df[(df['abs_error'] > 0.01) & (df['abs_error'] <= 1.00)]
print(f"\nCases within $0.01-$1.00 error: {len(near_misses)}")
if len(near_misses) > 0:
    print("\nSample near misses:")
    sample = near_misses.head(10)[['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'abs_error']]
    print(sample.to_string(index=False))

# Check for systematic bias
print("\n" + "="*60)
print("SYSTEMATIC BIAS ANALYSIS")
print("="*60)

df['error_signed'] = df['predicted'] - df['expected_output']
print(f"\nMean signed error: ${df['error_signed'].mean():.2f}")
print(f"Median signed error: ${df['error_signed'].median():.2f}")

over_predictions = (df['error_signed'] > 0).sum()
under_predictions = (df['error_signed'] < 0).sum()
print(f"\nOver-predictions: {over_predictions} ({over_predictions/len(df)*100:.1f}%)")
print(f"Under-predictions: {under_predictions} ({under_predictions/len(df)*100:.1f}%)")

# Analyze by magnitude
print("\nBias by reimbursement magnitude:")
df['magnitude_bin'] = pd.cut(df['expected_output'], bins=[0, 500, 1000, 1500, 2000, 5000])
bias_by_magnitude = df.groupby('magnitude_bin')['error_signed'].agg(['mean', 'std', 'count']).round(2)
print(bias_by_magnitude)

# Look for mathematical relationships
print("\n" + "="*60)
print("MATHEMATICAL RELATIONSHIP ANALYSIS")
print("="*60)

# Check if errors are proportional to magnitude
df['error_pct'] = df['error_signed'] / df['expected_output'] * 100
print(f"\nMean percentage error: {df['error_pct'].mean():.2f}%")
print(f"Std dev of percentage error: {df['error_pct'].std():.2f}%")

# Check for patterns in exact dollar amounts
print("\nChecking for rounding patterns...")
df['expected_rounded_5'] = (df['expected_output'] / 5).round() * 5
df['expected_rounded_10'] = (df['expected_output'] / 10).round() * 10
df['matches_5_rounding'] = abs(df['expected_output'] - df['expected_rounded_5']) < 0.01
df['matches_10_rounding'] = abs(df['expected_output'] - df['expected_rounded_10']) < 0.01

print(f"Expected values that are multiples of $5: {df['matches_5_rounding'].sum()}")
print(f"Expected values that are multiples of $10: {df['matches_10_rounding'].sum()}")

# Specific patterns in the 22 close matches
close_matches = df[df['abs_error'] <= 1.00]
print(f"\n{len(close_matches)} cases with error ≤ $1.00:")
if len(close_matches) > 0:
    print("\nPattern analysis of close matches:")
    # Check their characteristics
    print(f"  Average trip days: {close_matches['trip_days'].mean():.1f}")
    print(f"  Average miles: {close_matches['miles'].mean():.1f}")
    print(f"  Average receipts: ${close_matches['receipts'].mean():.2f}")
    
    # Check receipt endings
    close_matches['receipt_cents'] = (close_matches['receipts'] * 100).astype(int) % 100
    special_endings = close_matches['receipt_cents'].isin([49, 99]).sum()
    print(f"  With .49/.99 endings: {special_endings}")

# Final thought - are we missing a final transformation?
print("\n" + "="*60)
print("POTENTIAL MISSING TRANSFORMATIONS")
print("="*60)

# Check if a simple scaling factor would help
best_scale = df['expected_output'].sum() / df['predicted'].sum()
print(f"\nOptimal global scaling factor: {best_scale:.6f}")
df['predicted_scaled'] = df['predicted'] * best_scale
df['error_scaled'] = abs(df['predicted_scaled'] - df['expected_output'])
print(f"MAE with optimal scaling: ${df['error_scaled'].mean():.2f}")

# Check if there's a consistent offset
best_offset = df['error_signed'].median()
print(f"\nMedian offset: ${best_offset:.2f}")
df['predicted_offset'] = df['predicted'] - best_offset
df['error_offset'] = abs(df['predicted_offset'] - df['expected_output'])
print(f"MAE with offset correction: ${df['error_offset'].mean():.2f}") 