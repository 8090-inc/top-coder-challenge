"""Deep mathematical analysis to discover cents formula"""

import pandas as pd
import numpy as np
from collections import defaultdict
import math

def test_formula(df, formula_func, name):
    """Test a formula and return accuracy"""
    predicted = df.apply(formula_func, axis=1)
    matches = (predicted == df['output_cents']).sum()
    accuracy = matches / len(df) * 100
    return accuracy, matches, predicted

def analyze_cents_formula():
    """Try to discover the mathematical formula for cents"""
    
    # Load data
    df = pd.read_csv('public_cases_expected_outputs.csv')
    df['output_cents'] = (df['expected_output'] * 100).astype(int) % 100
    df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100
    df['output_dollars'] = df['expected_output'].astype(int)
    
    print("=== MATHEMATICAL FORMULA DISCOVERY ===")
    print(f"Total cases: {len(df)}")
    
    # Test various mathematical formulas
    formulas = {
        # Based on output dollars
        'dollars_mod_100': lambda row: row['output_dollars'] % 100,
        'dollars_mod_12': lambda row: (row['output_dollars'] * 12) % 100,
        'dollars_checksum': lambda row: sum(int(d) for d in str(row['output_dollars'])) % 100,
        
        # Based on inputs
        'input_sum_mod': lambda row: int(row['trip_days'] + row['miles'] + row['receipts']) % 100,
        'weighted_sum': lambda row: int(row['trip_days'] * 10 + row['miles'] * 0.1 + row['receipts']) % 100,
        'product_mod': lambda row: int(row['trip_days'] * row['miles']) % 100,
        
        # Complex formulas
        'complex1': lambda row: int((row['trip_days'] * 12 + row['miles'] + row['receipts'] * 100)) % 100,
        'complex2': lambda row: int((row['output_dollars'] + row['trip_days'] + int(row['miles'] / 10))) % 100,
        'complex3': lambda row: int((row['output_dollars'] * row['trip_days']) % 100),
        
        # Hash-like formulas
        'hash1': lambda row: int(abs(hash(f"{int(row['trip_days'])}{int(row['miles'])}{int(row['receipts']*100)}"))) % 100,
        'hash2': lambda row: int(abs(hash(f"{row['output_dollars']}{row['trip_days']}"))) % 100,
        
        # Based on clustering
        'cluster_based': lambda row: (
            12 if row['trip_days'] == 1 else
            24 if row['trip_days'] >= 10 else
            72 if 7 <= row['trip_days'] <= 8 else
            94
        ) if row['miles'] < 1000 else 33,
    }
    
    results = []
    
    print("\nTesting formulas:")
    for name, formula in formulas.items():
        accuracy, matches, predicted = test_formula(df, formula, name)
        results.append((accuracy, name, matches))
        print(f"  {name}: {matches}/{len(df)} ({accuracy:.1f}%)")
        
        if accuracy > 5:  # If better than random
            # Show distribution
            print(f"    Predicted cents distribution: {pd.Series(predicted).value_counts().head(5).to_dict()}")
    
    # Sort by accuracy
    results.sort(reverse=True)
    
    print("\n=== TOP FORMULAS ===")
    for accuracy, name, matches in results[:5]:
        print(f"{name}: {accuracy:.1f}% accuracy ({matches} matches)")
    
    # Analyze relationship between predicted amount and cents
    print("\n=== ANALYZING OUTPUT RELATIONSHIP ===")
    
    # Check if cents depend on the calculation before rounding
    # Simulate v5 model predictions
    from models.v5_practical_ensemble import calculate_reimbursement_v5
    
    print("\nChecking if cents relate to internal calculation...")
    v5_predictions = []
    for _, row in df.iterrows():
        pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
        v5_predictions.append(pred)
    
    df['v5_prediction'] = v5_predictions
    df['v5_cents'] = (df['v5_prediction'] * 100).astype(int) % 100
    
    # Check if output cents correlate with v5 cents
    v5_matches = (df['v5_cents'] == df['output_cents']).sum()
    print(f"V5 prediction cents match: {v5_matches}/{len(df)} ({v5_matches/len(df)*100:.1f}%)")
    
    # Analyze patterns for most common cents values
    print("\n=== PATTERN ANALYSIS FOR COMMON CENTS ===")
    common_cents = df['output_cents'].value_counts().head(10).index
    
    for cents in common_cents[:5]:
        subset = df[df['output_cents'] == cents]
        print(f"\nCents = {cents} ({len(subset)} cases):")
        
        # Check for patterns
        if len(subset) > 10:
            # Check dollar patterns
            dollar_pattern = subset['output_dollars'] % 100
            if len(dollar_pattern.unique()) <= 5:
                print(f"  Dollar pattern (mod 100): {dollar_pattern.value_counts().head().to_dict()}")
            
            # Check trip day patterns
            day_pattern = subset['trip_days'].value_counts().head(3)
            print(f"  Common trip days: {day_pattern.to_dict()}")
            
            # Check if there's a formula
            for divisor in [12, 24, 60, 100]:
                mod_pattern = (subset['output_dollars'] % divisor).value_counts()
                if len(mod_pattern) <= 3:
                    print(f"  Output dollars % {divisor} = {mod_pattern.to_dict()}")
    
    # Look for exact mathematical relationships
    print("\n=== SEARCHING FOR EXACT RELATIONSHIPS ===")
    
    # Group by cents and look for patterns
    for cents in [12, 24, 72, 94]:  # Most common
        subset = df[df['output_cents'] == cents]
        
        # Check if there's a simple relationship
        for i in range(1, 10):
            for j in range(1, 10):
                # Test formula: (output_dollars * i + trip_days * j) % 100 == cents
                test = ((subset['output_dollars'] * i + subset['trip_days'] * j) % 100 == cents).all()
                if test and len(subset) > 5:
                    print(f"FOUND: cents {cents} = (dollars * {i} + days * {j}) % 100")
                    
                # Test with miles
                test2 = ((subset['output_dollars'] * i + subset['miles'] / j) % 100).astype(int)
                if (test2 == cents).all() and len(subset) > 5:
                    print(f"FOUND: cents {cents} = (dollars * {i} + miles / {j}) % 100")

if __name__ == "__main__":
    analyze_cents_formula() 