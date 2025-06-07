#!/usr/bin/env python3
"""
Comprehensive Analysis of ACME Reimbursement Engine
Analyzes patterns in public_cases.json to reverse engineer the algorithm
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_data():
    """Load public cases and return DataFrame"""
    with open('../public_cases.json', 'r') as f:
        data = json.load(f)
    
    rows = []
    for i, case in enumerate(data):
        inp = case['input']
        rows.append({
            'case_id': i,
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled'],
            'receipts': inp['total_receipts_amount'],
            'expected': case['expected_output'],
            'mpd': inp['miles_traveled'] / inp['trip_duration_days']
        })
    
    return pd.DataFrame(rows)

def analyze_by_days(df):
    """Analyze patterns by trip duration"""
    print("=" * 60)
    print("TRIP DURATION ANALYSIS")
    print("=" * 60)
    
    for days in sorted(df['days'].unique()):
        subset = df[df['days'] == days]
        avg_total = subset['expected'].mean()
        avg_per_day = avg_total / days
        
        print(f"\n{days} day trips (n={len(subset)}):")
        print(f"  Total avg: ${avg_total:.2f}")
        print(f"  Per-day avg: ${avg_per_day:.2f}")
        print(f"  Range: ${subset['expected'].min():.2f} - ${subset['expected'].max():.2f}")
        
        # Correlation analysis
        if len(subset) > 5:
            miles_corr = subset[['miles', 'expected']].corr().iloc[0,1]
            receipts_corr = subset[['receipts', 'expected']].corr().iloc[0,1]
            mpd_corr = subset[['mpd', 'expected']].corr().iloc[0,1]
            
            print(f"  Correlations - Miles: {miles_corr:.3f}, Receipts: {receipts_corr:.3f}, MPD: {mpd_corr:.3f}")

def analyze_component_contributions(df):
    """Try to reverse engineer component contributions"""
    print("\n" + "=" * 60)
    print("COMPONENT CONTRIBUTION ANALYSIS")  
    print("=" * 60)
    
    # Analyze cases with minimal receipts to isolate base + mileage
    low_receipt_cases = df[df['receipts'] < 10]
    print(f"\nLow receipt cases (n={len(low_receipt_cases)}):")
    
    for days in sorted(low_receipt_cases['days'].unique()):
        subset = low_receipt_cases[low_receipt_cases['days'] == days]
        if len(subset) > 2:
            # Try to find base rate by looking at zero/low mileage cases
            low_miles = subset[subset['miles'] < 20]
            if len(low_miles) > 0:
                estimated_base = low_miles['expected'].mean()
                print(f"  {days} days - Estimated base: ${estimated_base:.2f}")
            
            # Estimate mileage rate
            if len(subset) > 3:
                # Linear regression on miles vs expected
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(subset['miles'], subset['expected'])
                print(f"    Mileage rate: ${slope:.4f}/mile (R²={r_value**2:.3f})")

def analyze_receipt_patterns(df):
    """Analyze receipt processing patterns"""
    print("\n" + "=" * 60)
    print("RECEIPT PROCESSING ANALYSIS")
    print("=" * 60)
    
    # Create receipt buckets
    receipt_buckets = [0, 10, 50, 100, 200, 500, 1000, 2000, float('inf')]
    bucket_labels = ['0-10', '10-50', '50-100', '100-200', '200-500', '500-1000', '1000-2000', '2000+']
    
    df['receipt_bucket'] = pd.cut(df['receipts'], bins=receipt_buckets, labels=bucket_labels, right=False)
    
    for bucket in bucket_labels:
        subset = df[df['receipt_bucket'] == bucket]
        if len(subset) > 0:
            # Calculate apparent receipt multiplier
            # This is rough but gives us a sense
            non_receipt_estimate = subset['days'] * 100 + subset['miles'] * 0.3
            receipt_component = subset['expected'] - non_receipt_estimate
            apparent_multiplier = receipt_component / subset['receipts']
            
            print(f"\n{bucket} receipts (n={len(subset)}):")
            print(f"  Avg receipts: ${subset['receipts'].mean():.2f}")
            print(f"  Avg expected: ${subset['expected'].mean():.2f}")
            print(f"  Apparent multiplier: {apparent_multiplier.mean():.2f}x (±{apparent_multiplier.std():.2f})")

def find_outliers_and_patterns(df):
    """Find unusual cases that might reveal special logic"""
    print("\n" + "=" * 60)
    print("OUTLIER AND PATTERN ANALYSIS")
    print("=" * 60)
    
    # High MPD cases
    high_mpd = df[df['mpd'] > 200]
    print(f"\nHigh MPD cases (>200 mpd, n={len(high_mpd)}):")
    for _, row in high_mpd.head(10).iterrows():
        print(f"  Case {row['case_id']}: {row['days']}d, {row['miles']}mi, ${row['receipts']:.2f} → ${row['expected']:.2f} ({row['mpd']:.1f} mpd)")
    
    # 1-day high mile cases (our main problem)
    one_day_high_miles = df[(df['days'] == 1) & (df['miles'] > 500)]
    print(f"\n1-day high mileage cases (>500mi, n={len(one_day_high_miles)}):")
    for _, row in one_day_high_miles.head(10).iterrows():
        estimated_simple = 120 + row['miles'] * 0.3 + row['receipts'] * 0.5
        print(f"  Case {row['case_id']}: {row['miles']}mi, ${row['receipts']:.2f} → Expected: ${row['expected']:.2f}, Simple: ${estimated_simple:.2f}")

def analyze_windfall_patterns(df):
    """Look for windfall patterns (.49/.99 endings)"""
    print("\n" + "=" * 60)
    print("WINDFALL PATTERN ANALYSIS")
    print("=" * 60)
    
    # Check for .49 and .99 receipt endings
    df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100
    windfall_cases = df[df['receipt_cents'].isin([49, 99])]
    normal_cases = df[~df['receipt_cents'].isin([49, 99])]
    
    print(f"Windfall cases (.49/.99): {len(windfall_cases)}")
    print(f"Normal cases: {len(normal_cases)}")
    
    if len(windfall_cases) > 10:
        print(f"Avg windfall payout: ${windfall_cases['expected'].mean():.2f}")
        print(f"Avg normal payout: ${normal_cases['expected'].mean():.2f}")

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} test cases")
    
    analyze_by_days(df)
    analyze_component_contributions(df)
    analyze_receipt_patterns(df)
    find_outliers_and_patterns(df)
    analyze_windfall_patterns(df)
    
    # Save detailed analysis
    print("\nSaving detailed case analysis...")
    df.to_csv('../analysis_detailed.csv', index=False)
    print("Analysis complete. Check analysis_detailed.csv for full data.")

if __name__ == "__main__":
    main() 