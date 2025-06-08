"""Comprehensive cents formula discovery - handling all edge cases"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
sys.path.append('.')

def safe_checksum(value):
    """Calculate checksum safely"""
    return sum(int(d) for d in str(int(value)))

def analyze_cents_patterns_comprehensive():
    """Comprehensive analysis to discover cents pattern"""
    
    # Load data
    df = pd.read_csv('public_cases_expected_outputs.csv')
    df['output_cents'] = (df['expected_output'] * 100).astype(int) % 100
    df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100
    df['output_dollars'] = df['expected_output'].astype(int)
    
    print("=== COMPREHENSIVE CENTS ANALYSIS ===")
    print(f"Total cases: {len(df)}")
    print(f"Unique output cents: {df['output_cents'].nunique()}")
    
    # First, let's check if the cents are actually from the v5 model's internal calculation
    print("\n1. CHECKING V5 MODEL CALCULATION...")
    from models.v5_practical_ensemble import calculate_reimbursement_v5
    
    correct_predictions = 0
    cent_matches = 0
    
    for idx, row in df.iterrows():
        v5_pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
        
        # Check exact match
        if abs(v5_pred - row['expected_output']) < 0.01:
            correct_predictions += 1
            
        # Check cents match
        v5_cents = int(v5_pred * 100) % 100
        if v5_cents == row['output_cents']:
            cent_matches += 1
    
    print(f"V5 exact matches: {correct_predictions}/{len(df)}")
    print(f"V5 cents matches: {cent_matches}/{len(df)} ({cent_matches/len(df)*100:.1f}%)")
    
    # Analyze the pattern of cents
    print("\n2. CENTS DISTRIBUTION ANALYSIS...")
    print("Top 20 most common cents:")
    print(df['output_cents'].value_counts().head(20))
    
    # Look for patterns based on calculation rules
    print("\n3. RULE-BASED PATTERN ANALYSIS...")
    
    # Get cluster assignments from v3 model
    from models.cluster_router import assign_cluster
    
    df['cluster'] = df.apply(lambda row: assign_cluster(row['trip_days'], row['miles'], row['receipts']), axis=1)
    
    # Analyze cents by cluster
    print("\nCents distribution by cluster:")
    for cluster in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster]
        top_cents = subset['output_cents'].value_counts().head(5)
        print(f"\nCluster {cluster} ({len(subset)} cases):")
        print(f"  Top cents: {top_cents.to_dict()}")
    
    # Check if cents relate to pre-penalty amount
    print("\n4. PRE-PENALTY AMOUNT ANALYSIS...")
    
    # For receipts ending in .49 and .99, reverse the penalty
    df['pre_penalty_amount'] = df.apply(lambda row: (
        row['expected_output'] / 0.341 if row['receipt_cents'] == 49 else
        row['expected_output'] / 0.51 if row['receipt_cents'] == 99 else
        row['expected_output']
    ), axis=1)
    
    df['pre_penalty_cents'] = (df['pre_penalty_amount'] * 100).astype(int) % 100
    
    # Check patterns in pre-penalty cents
    print("\nPre-penalty cents for .49 and .99 receipts:")
    for ending in [49, 99]:
        mask = df['receipt_cents'] == ending
        if mask.sum() > 0:
            subset = df[mask]
            print(f"\n.{ending} receipts ({mask.sum()} cases):")
            print(f"  Output cents: {subset['output_cents'].value_counts().head(5).to_dict()}")
            print(f"  Pre-penalty cents: {subset['pre_penalty_cents'].value_counts().head(5).to_dict()}")
    
    # Mathematical formula testing
    print("\n5. MATHEMATICAL FORMULA TESTING...")
    
    # Test if cents come from a calculation involving all inputs
    test_results = []
    
    # Test various formulas
    for a in range(0, 10):
        for b in range(0, 10):
            for c in range(0, 5):
                # Formula: (a * days + b * miles/100 + c * receipts) % 100
                predicted_cents = ((a * df['trip_days'] + b * df['miles']/100 + c * df['receipts']).astype(int) % 100)
                matches = (predicted_cents == df['output_cents']).sum()
                if matches > 50:  # If better than 5%
                    test_results.append((matches, f"({a}*days + {b}*miles/100 + {c}*receipts) % 100"))
    
    # Sort by matches
    test_results.sort(reverse=True)
    
    print("\nTop formula candidates:")
    for matches, formula in test_results[:10]:
        print(f"  {formula}: {matches}/{len(df)} matches ({matches/len(df)*100:.1f}%)")
    
    # Analyze specific cent values
    print("\n6. SPECIFIC CENTS PATTERN ANALYSIS...")
    
    # For the most common cents, look for deterministic patterns
    common_cents = [24, 12, 94, 72, 16, 68, 34, 33, 96, 18]
    
    for cents in common_cents[:5]:
        subset = df[df['output_cents'] == cents]
        print(f"\nAnalyzing cents = {cents} ({len(subset)} cases):")
        
        # Check if all cases share common characteristics
        if len(subset) > 10:
            # Check modulo patterns
            days_mod_7 = subset['trip_days'] % 7
            if len(days_mod_7.value_counts()) <= 3:
                print(f"  days % 7 pattern: {days_mod_7.value_counts().to_dict()}")
            
            # Check if related to output magnitude
            output_range = (subset['expected_output'].min(), subset['expected_output'].max())
            print(f"  Output range: ${output_range[0]:.2f} - ${output_range[1]:.2f}")
            
            # Check cluster distribution
            cluster_dist = subset['cluster'].value_counts()
            print(f"  Cluster distribution: {cluster_dist.to_dict()}")
    
    # Final hypothesis testing
    print("\n7. TESTING FINAL HYPOTHESES...")
    
    # Hypothesis 1: Cents are deterministic based on cluster + some calculation
    print("\nHypothesis 1: Cluster-based deterministic cents")
    for cluster in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster]
        # Check if this cluster has a dominant cents pattern
        top_cent = subset['output_cents'].mode()
        if len(top_cent) > 0:
            coverage = (subset['output_cents'] == top_cent.iloc[0]).sum() / len(subset)
            if coverage > 0.3:  # If 30%+ have same cents
                print(f"  Cluster {cluster}: {coverage*100:.1f}% have cents = {top_cent.iloc[0]}")
    
    # Hypothesis 2: Cents come from truncation/rounding artifacts
    print("\nHypothesis 2: Truncation artifacts")
    # Calculate what cents would be if we truncated instead of rounded
    df['truncated_output'] = np.floor(df['expected_output'] * 100) / 100
    df['truncated_cents'] = (df['truncated_output'] * 100).astype(int) % 100
    truncation_matches = (df['truncated_cents'] == df['output_cents']).sum()
    print(f"  Truncation matches: {truncation_matches}/{len(df)} ({truncation_matches/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    df = analyze_cents_patterns_comprehensive() 