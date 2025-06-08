#!/usr/bin/env python3
"""
Analyze outlier cases with high prediction errors from the v4 model
"""

import pandas as pd
import numpy as np

# Load the v4 predictions
df = pd.read_csv('public_cases_predictions_v4.csv')

# Sort by absolute error
df = df.sort_values('abs_error', ascending=False)

# Analyze top outliers (errors > $100)
outliers = df[df['abs_error'] > 100].copy()

print(f"Total cases with error > $100: {len(outliers)}")
print(f"Maximum error: ${df['abs_error'].max():.2f}")
print("\n" + "="*80 + "\n")

# Add analysis columns
outliers['error_direction'] = outliers['error'].apply(lambda x: 'over' if x > 0 else 'under')
outliers['receipts_cents'] = (outliers['receipts'] * 100).astype(int) % 100
outliers['has_49_ending'] = outliers['receipts_cents'] == 49
outliers['has_99_ending'] = outliers['receipts_cents'] == 99
outliers['miles_per_day'] = outliers['miles'] / outliers['trip_days']
outliers['receipts_per_day'] = outliers['receipts'] / outliers['trip_days']

# Display all outliers with details
print("All cases with error > $100:")
print("-" * 80)
for idx, row in outliers.iterrows():
    print(f"\nCase: {row['trip_days']} days, {row['miles']:.0f} miles, ${row['receipts']:.2f} receipts")
    print(f"Expected: ${row['expected_output']:.2f}, Predicted: ${row['predicted']:.2f}")
    print(f"Error: ${row['abs_error']:.2f} ({row['error_direction']}-predicted)")
    print(f"Receipt ending: {row['receipts_cents']:02d} ({'Special' if row['has_49_ending'] or row['has_99_ending'] else 'Normal'})")
    print(f"Miles/day: {row['miles_per_day']:.1f}, Receipts/day: ${row['receipts_per_day']:.2f}")

# Look for patterns
print("\n" + "="*80 + "\n")
print("PATTERN ANALYSIS:")
print("-" * 80)

# Receipt endings
print(f"\nReceipt endings in outliers:")
ending_counts = outliers['receipts_cents'].value_counts().head(10)
print(ending_counts)

# Trip duration patterns
print(f"\nTrip duration distribution:")
print(outliers['trip_days'].value_counts().sort_index())

# Miles patterns
print(f"\nMiles statistics:")
print(f"Mean: {outliers['miles'].mean():.1f}, Median: {outliers['miles'].median():.1f}")
print(f"Min: {outliers['miles'].min():.1f}, Max: {outliers['miles'].max():.1f}")

# Receipts patterns  
print(f"\nReceipts statistics:")
print(f"Mean: ${outliers['receipts'].mean():.2f}, Median: ${outliers['receipts'].median():.2f}")
print(f"Min: ${outliers['receipts'].min():.2f}, Max: ${outliers['receipts'].max():.2f}")

# Low vs high receipts
print(f"\nLow receipts (<$50): {(outliers['receipts'] < 50).sum()} cases")
print(f"Very high receipts (>$2000): {(outliers['receipts'] > 2000).sum()} cases")

# Direction of errors
print(f"\nError direction:")
print(outliers['error_direction'].value_counts())

# Efficiency patterns
print(f"\nEfficiency patterns:")
print(f"Low miles/day (<50): {(outliers['miles_per_day'] < 50).sum()} cases")
print(f"Very high miles/day (>400): {(outliers['miles_per_day'] > 400).sum()} cases")

# Look for specific patterns
print("\n" + "="*80 + "\n")
print("SPECIFIC PATTERN INSIGHTS:")
print("-" * 80)

# Pattern 1: Very low receipts
low_receipt_outliers = outliers[outliers['receipts'] < 50]
if len(low_receipt_outliers) > 0:
    print(f"\n1. Very low receipts (<$50): {len(low_receipt_outliers)} cases")
    print("   These cases have receipts:", low_receipt_outliers['receipts'].tolist())

# Pattern 2: .49 or .99 endings
special_ending_outliers = outliers[outliers['has_49_ending'] | outliers['has_99_ending']]
if len(special_ending_outliers) > 0:
    print(f"\n2. Special receipt endings (.49/.99): {len(special_ending_outliers)} cases")
    for _, row in special_ending_outliers.iterrows():
        print(f"   - {row['trip_days']}d, {row['miles']:.0f}mi, ${row['receipts']:.2f} → Error ${row['abs_error']:.2f}")

# Pattern 3: Very high receipts with low miles
high_receipt_low_mile = outliers[(outliers['receipts'] > 2000) & (outliers['miles'] < 100)]
if len(high_receipt_low_mile) > 0:
    print(f"\n3. High receipts + low miles: {len(high_receipt_low_mile)} cases")
    for _, row in high_receipt_low_mile.iterrows():
        print(f"   - {row['trip_days']}d, {row['miles']:.0f}mi, ${row['receipts']:.2f} → Error ${row['abs_error']:.2f}")

# Pattern 4: Specific day patterns
for days in [2, 4, 7, 11]:
    day_outliers = outliers[outliers['trip_days'] == days]
    if len(day_outliers) > 0:
        print(f"\n4. {days}-day trips: {len(day_outliers)} outliers")
        print(f"   Average error: ${day_outliers['abs_error'].mean():.2f}")

# Find which clusters these outliers belong to
print("\n" + "="*80 + "\n")
print("CLUSTER ASSIGNMENT FOR OUTLIERS:")
print("-" * 80)

def assign_cluster(row):
    """Assign cluster based on v3 rules"""
    days = row['trip_days']
    miles = row['miles']
    receipts = row['receipts']
    
    if days == 1:
        if miles >= 600:
            if receipts >= 800:
                return '1a'
            else:
                return '1b'
        else:
            return '6'
    elif days < 100 and miles < 200 and receipts > 1500:
        return '0_low_mile_high_receipt'
    elif days >= 10 and receipts >= 1100:
        return '2'
    elif 3 <= days <= 5 and receipts >= 1400:
        return '3'
    elif receipts <= 24:
        return '4'
    elif 5 <= days <= 12 and miles >= 700:
        if 7 <= days <= 8 and 900 <= miles <= 1200 and 1000 <= receipts <= 1200:
            return '5_special'
        else:
            return '5'
    else:
        return '0'

outliers['cluster'] = outliers.apply(assign_cluster, axis=1)
print("\nCluster distribution of outliers:")
print(outliers['cluster'].value_counts())

# Show worst case from each cluster
print("\nWorst case from each cluster:")
for cluster in outliers['cluster'].unique():
    worst = outliers[outliers['cluster'] == cluster].iloc[0]
    print(f"\nCluster {cluster}:")
    print(f"  {worst['trip_days']}d, {worst['miles']:.0f}mi, ${worst['receipts']:.2f}")
    print(f"  Error: ${worst['abs_error']:.2f} ({worst['error_direction']}-predicted)") 