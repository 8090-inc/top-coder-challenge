"""Analyze cents patterns in expected outputs to discover the formula"""

import pandas as pd
import numpy as np
from collections import Counter
import math

def analyze_cents_patterns():
    """Analyze the cents patterns in expected outputs"""
    
    # Load data
    df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Extract cents
    df['output_cents'] = (df['expected_output'] * 100).astype(int) % 100
    df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100
    
    print("=== CENTS PATTERN ANALYSIS ===")
    print(f"Total cases: {len(df)}")
    print(f"Unique cents values: {df['output_cents'].nunique()}")
    
    # Top cents patterns
    print("\nTop 20 most common output cents:")
    print(df['output_cents'].value_counts().head(20))
    
    # Check if 988/1000 have specific patterns
    common_cents = [12, 24, 72, 94, 16, 68, 34, 33, 96, 18]
    common_count = df[df['output_cents'].isin(common_cents)].shape[0]
    print(f"\nCases with top 10 common cents: {common_count} / {len(df)}")
    
    # Test various theories
    print("\n=== TESTING CENTS THEORIES ===")
    
    theories = {}
    
    # Theory 1: Simple modulo of sum
    df['theory1'] = (df['trip_days'] + df['miles'] + df['receipts']).astype(int) % 100
    
    # Theory 2: Product modulo
    df['theory2'] = (df['trip_days'] * df['miles']).astype(int) % 100
    
    # Theory 3: Checksum approach
    def checksum(row):
        digits = str(int(row['trip_days'])) + str(int(row['miles'])) + str(int(row['receipts'] * 100))
        return sum(int(d) for d in digits) % 100
    df['theory3'] = df.apply(checksum, axis=1)
    
    # Theory 4: Based on receipt cents
    df['theory4'] = (df['receipt_cents'] + df['trip_days']) % 100
    
    # Theory 5: Hash-based mapping
    def hash_theory(row):
        known_cents = [12, 24, 72, 94, 16, 68, 34, 33, 96, 18]
        hash_val = hash(f"{int(row['trip_days'])}{int(row['miles'])}{int(row['receipts']*100)}")
        return known_cents[abs(hash_val) % len(known_cents)]
    df['theory5'] = df.apply(hash_theory, axis=1)
    
    # Theory 6: Complex formula
    df['theory6'] = ((df['trip_days'] * 12) + (df['miles'] % 100) + df['receipt_cents']) % 100
    
    # Evaluate theories
    for theory in ['theory1', 'theory2', 'theory3', 'theory4', 'theory5', 'theory6']:
        matches = (df[theory] == df['output_cents']).sum()
        accuracy = matches / len(df) * 100
        print(f"\n{theory}: {matches}/{len(df)} matches ({accuracy:.1f}%)")
        
        # Show some examples where it works
        if matches > 0:
            print("  Examples where it works:")
            working = df[df[theory] == df['output_cents']].head(3)
            for _, row in working.iterrows():
                print(f"    Days={row['trip_days']}, Miles={row['miles']}, Receipts={row['receipts']:.2f} -> {row['output_cents']}")
    
    # Pattern analysis
    print("\n=== PATTERN ANALYSIS ===")
    
    # Group by output cents and look for input patterns
    for cents in [12, 24, 72, 94]:  # Most common
        subset = df[df['output_cents'] == cents]
        if len(subset) > 5:
            print(f"\nOutput cents = {cents} ({len(subset)} cases):")
            
            # Check for patterns in inputs
            print(f"  Trip days mode: {subset['trip_days'].mode().values[0]}")
            print(f"  Receipt cents common: {subset['receipt_cents'].value_counts().head(3).to_dict()}")
            
            # Check modulo patterns
            for mod in [7, 12, 24, 30]:
                days_mod = subset['trip_days'] % mod
                if len(days_mod.mode()) > 0 and (days_mod == days_mod.mode().values[0]).sum() > len(subset) * 0.5:
                    print(f"  Pattern: trip_days % {mod} = {days_mod.mode().values[0]} in {(days_mod == days_mod.mode().values[0]).sum()}/{len(subset)} cases")

    # Special cases analysis
    print("\n=== SPECIAL CASES ===")
    # Look at .49 and .99 receipt endings
    for ending in [49, 99]:
        mask = df['receipt_cents'] == ending
        if mask.sum() > 0:
            print(f"\nReceipts ending in .{ending}: {mask.sum()} cases")
            print(f"  Output cents distribution: {df[mask]['output_cents'].value_counts().head(5).to_dict()}")

if __name__ == "__main__":
    analyze_cents_patterns() 