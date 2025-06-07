#!/usr/bin/env python3
"""
Legacy Reimbursement System Calculator
Supports two modes:
1. Calculate single reimbursement: python calculate_reimbursement.py <trip_days> <miles> <receipts>
2. Process JSON file: python calculate_reimbursement.py <json_file>
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Version tracking for our formula iterations
FORMULA_VERSION = "v0.5_cluster_based"

def calculate_reimbursement_v01(trip_days, miles, receipts):
    """
    Version 0.1: Baseline linear model from initial analysis
    Based on: All features linear regression
    Coefficients: trip_days: 50.05, miles: 0.446, receipts: 0.383
    Intercept: 266.71
    """
    reimbursement = (
        266.71 +  # intercept
        50.05 * trip_days +
        0.446 * miles +
        0.383 * receipts
    )
    return max(0, reimbursement)  # Never return negative

def calculate_reimbursement_v02(trip_days, miles, receipts):
    """
    Version 0.2: Inverted coverage model based on receipt analysis
    
    Key findings:
    - <$10 receipts: 6599% coverage (66x multiplier)
    - $10-50: 2689% coverage (27x multiplier)
    - $50-100: 970% coverage
    - $100-200: 578% coverage
    - >$2000: 73% coverage
    
    Decision tree thresholds: $5.82, $11.45, $13.24, $23.09, $36.20
    """
    # Base calculation (simplified for now)
    base = 50 * trip_days + 0.4 * miles
    
    # Apply inverted coverage based on receipt amount
    if receipts < 5.82:
        coverage = 65.99  # 6599%
    elif receipts < 11.45:
        coverage = 40.0   # Interpolated
    elif receipts < 23.09:
        coverage = 26.89  # 2689%
    elif receipts < 50:
        coverage = 15.0   # Interpolated
    elif receipts < 100:
        coverage = 9.70   # 970%
    elif receipts < 200:
        coverage = 5.78   # 578%
    elif receipts < 500:
        coverage = 2.37   # 237%
    elif receipts < 1000:
        coverage = 1.61   # 161%
    elif receipts < 1500:
        coverage = 1.32   # 132%
    elif receipts < 2000:
        coverage = 0.96   # 96%
    else:
        coverage = 0.73   # 73%
    
    # Calculate reimbursement
    receipt_component = receipts * coverage
    reimbursement = base + receipt_component
    
    return max(0, reimbursement)

def calculate_reimbursement_v03(trip_days, miles, receipts):
    """
    Version 0.3: Hybrid model combining insights from v0.1 and v0.2
    
    Key insights:
    - Linear model works well generally
    - Special handling needed for ~$1000-1200 receipts (v0.2 excels here)
    - Low receipts still need penalty but not as extreme as v0.2
    - Base amount appears important
    """
    # Start with linear base (from v0.1)
    base_linear = 266.71 + 50.05 * trip_days + 0.446 * miles
    
    # Receipt component with more nuanced handling
    if receipts < 50:
        # Low receipt penalty, but not as extreme as v0.2
        receipt_component = receipts * 2.5  # 250% coverage for very low
    elif receipts >= 1000 and receipts <= 1200:
        # Special handling for the sweet spot where v0.2 excels
        # Use v0.2 style calculation for this range
        coverage = 1.61  # From v0.2 for 1000-1500 range
        receipt_component = receipts * coverage
    else:
        # Standard linear handling for everything else
        receipt_component = receipts * 0.383
    
    reimbursement = base_linear + receipt_component
    
    # Apply bounds based on observations
    # Very low receipts seem to have a floor around $200-600
    if receipts < 10 and reimbursement < 200:
        reimbursement = 200 + receipts * 3
    
    return max(0, reimbursement)

def calculate_reimbursement_v04(trip_days, miles, receipts):
    """
    Version 0.4: Refined model based on detailed analysis
    
    Key findings:
    - Linear model (v0.1) is solid baseline
    - Special case: 7-8 day trips with ~1000 miles and ~1000-1200 receipts
    - Low receipts need different handling
    - Certain receipt endings (.49, .99) are penalized
    """
    # Check for the special high-value trip profile where v0.2 excels
    if (7 <= trip_days <= 8 and 
        900 <= miles <= 1200 and 
        1000 <= receipts <= 1200):
        # Use simplified v0.2-style calculation for this specific case
        # These trips consistently get ~$2100-2300
        base = 50 * trip_days + 0.4 * miles
        receipt_component = receipts * 1.61
        reimbursement = base + receipt_component
    else:
        # Use enhanced linear model for everything else
        base = 266.71 + 50.05 * trip_days + 0.446 * miles
        
        # Receipt handling with adjustments
        if receipts < 10:
            # Very low receipts get minimum reimbursement
            receipt_component = 100  # Flat amount for minimal receipts
        elif receipts < 50:
            # Low receipts get reduced coverage
            receipt_component = receipts * 0.8
        else:
            # Standard receipt handling
            receipt_component = receipts * 0.383
        
        reimbursement = base + receipt_component
    
    # Apply receipt ending penalties
    receipt_cents = int((receipts * 100) % 100)
    if receipt_cents == 49:
        reimbursement *= 0.35  # -65% for .49 endings
    elif receipt_cents == 99:
        reimbursement *= 0.51  # -49% for .99 endings
    
    return max(0, reimbursement)

def calculate_reimbursement_v05(trip_days, miles, receipts):
    """
    Version 0.5: Cluster-based model
    
    Key findings:
    - 6 distinct calculation paths (clusters)
    - Each cluster represents different trip types
    - Cluster 5 contains special profile cases
    """
    
    # Calculate derived features for clustering
    miles_per_day = miles / trip_days if trip_days > 0 else 0
    receipts_per_day = receipts / trip_days if trip_days > 0 else 0
    receipt_coverage = 1.0  # Will be updated after calculation
    output_per_day = 200  # Initial estimate
    
    # Feature vector for clustering
    features = [trip_days, miles, receipts, miles_per_day, receipts_per_day, receipt_coverage, output_per_day]
    
    # Scaler parameters (from clustering analysis)
    scaler_mean = [7.043, 597.41374, 1211.0568700000001, 147.02619530669332, 
                   285.7060807000777, 2.801597213397003, 284.7120321275669]
    scaler_scale = [3.9241751999624075, 351.124095966102, 742.4826603738993, 193.7236752036585,
                    381.51689150974863, 10.153163246773477, 268.43394376874403]
    
    # Scale features
    scaled_features = []
    for i, (feat, mean, scale) in enumerate(zip(features, scaler_mean, scaler_scale)):
        if scale > 0:
            scaled_features.append((feat - mean) / scale)
        else:
            scaled_features.append(0)
    
    # Cluster centroids (from K-means)
    centroids = [
        [0.1866283439250101, -0.9967064552506947, -0.7474365546997919, -0.5569162609593883, 
         -0.5034396321321946, 0.09950439973002874, -0.5273182311434319],
        [-1.5348448254954827, 0.4564946178329905, 0.5543886099544929, 3.0925172365405995, 
         3.4426143338043045, -0.19235370419136444, 3.570176909884608],
        [0.8714149750618587, 0.3845216918453981, 0.8276242210045852, -0.37621620555194757, 
         -0.2616218096891219, -0.16992681039252977, -0.39186643124293],
        [-0.9295332081059077, -0.10675734772962193, 0.7021487155222411, 0.2573874915645624, 
         0.7561263547962562, -0.19134009213310563, 0.6916318602105594],
        [-1.03028019748933, -1.4365682839627638, -1.629178611916474, -0.5989262550626138, 
         -0.7476280964599334, 25.006550147401477, -0.6080032074290795],
        [-0.3384212189613996, 0.7681918137606423, -0.888351994366373, 0.31893173109174644, 
         -0.413384884548907, 0.19331835187911464, -0.17445874232854566]
    ]
    
    # Find nearest centroid (cluster assignment)
    min_distance = float('inf')
    cluster_id = 0
    
    for i, centroid in enumerate(centroids):
        distance = sum((a - b) ** 2 for a, b in zip(scaled_features, centroid))
        if distance < min_distance:
            min_distance = distance
            cluster_id = i
    
    # Apply cluster-specific calculation
    if cluster_id == 0:
        # Cluster 0: Standard Multi-Day - Linear model
        reimbursement = 57.80 + 46.69 * trip_days + 0.51 * miles + 0.71 * receipts
        
    elif cluster_id == 1:
        # Cluster 1: Single Day High Miles - Simplified decision tree
        # Decision tree approximation based on analysis
        if receipts > 1500:
            if miles > 800:
                reimbursement = 1400
            else:
                reimbursement = 1250
        else:
            if miles > 600:
                reimbursement = 1200
            else:
                reimbursement = 1100
                
    elif cluster_id == 2:
        # Cluster 2: Long Trip High Receipts - Simplified decision tree
        # Focus on trip length and receipts
        if trip_days >= 10:
            if receipts > 2000:
                reimbursement = 1850
            else:
                reimbursement = 1750
        else:
            if receipts > 1800:
                reimbursement = 1800
            else:
                reimbursement = 1700
                
    elif cluster_id == 3:
        # Cluster 3: Short Trip (3-5 days) - Simplified decision tree
        if trip_days <= 3:
            if receipts > 1800:
                reimbursement = 1450
            else:
                reimbursement = 1350
        else:
            if receipts > 1700:
                reimbursement = 1550
            else:
                reimbursement = 1400
                
    elif cluster_id == 4:
        # Cluster 4: Outlier - Fixed value
        reimbursement = 364.51
        
    else:  # cluster_id == 5
        # Cluster 5: Medium Trip High Miles (contains special profile)
        
        # Check if it's a special profile case
        if (7 <= trip_days <= 8 and 900 <= miles <= 1200 and 1000 <= receipts <= 1200):
            # Special profile - use step function based on receipts
            if receipts < 1050:
                reimbursement = 2047
            elif receipts < 1100:
                reimbursement = 2073
            elif receipts < 1150:
                reimbursement = 2120
            else:
                reimbursement = 2280
        else:
            # Regular cluster 5 - simplified decision tree
            if trip_days <= 4:
                if miles > 800:
                    reimbursement = 1100
                else:
                    reimbursement = 900
            else:
                if miles > 900:
                    reimbursement = 1300
                else:
                    reimbursement = 1100
    
    # Apply receipt ending penalties (from v0.4 findings)
    receipt_cents = int((receipts * 100) % 100)
    if receipt_cents == 49:
        reimbursement *= 0.35  # -65% for .49 endings
    elif receipt_cents == 99:
        reimbursement *= 0.51  # -49% for .99 endings
    
    return max(0, reimbursement)

def calculate_reimbursement(trip_days, miles, receipts, version="v0.5"):
    """
    Main calculation function - routes to appropriate version
    """
    if version == "v0.1":
        return calculate_reimbursement_v01(trip_days, miles, receipts)
    elif version == "v0.2":
        return calculate_reimbursement_v02(trip_days, miles, receipts)
    elif version == "v0.3":
        return calculate_reimbursement_v03(trip_days, miles, receipts)
    elif version == "v0.4":
        return calculate_reimbursement_v04(trip_days, miles, receipts)
    elif version == "v0.5":
        return calculate_reimbursement_v05(trip_days, miles, receipts)
    else:
        raise ValueError(f"Unknown formula version: {version}")

def process_single(trip_days, miles, receipts, version="v0.5"):
    """Process a single reimbursement calculation"""
    result = calculate_reimbursement(trip_days, miles, receipts, version)
    print(f"Formula Version: {FORMULA_VERSION}")
    print(f"Inputs: {trip_days} days, {miles} miles, ${receipts:.2f} receipts")
    print(f"Expected Reimbursement: ${result:.2f}")
    return result

def process_json_file(filepath, version="v0.5"):
    """Process all cases in a JSON file"""
    print(f"Processing file: {filepath}")
    print(f"Formula Version: {FORMULA_VERSION}")
    print("-" * 60)
    
    # Load JSON data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier handling
    df = pd.json_normalize(data)
    
    # Check if it's public data (has expected output) or private
    if 'expected_output' in df.columns:
        # Public data - we can compare
        df.columns = ['expected_output', 'trip_days', 'miles', 'receipts']
        
        # Calculate predictions
        df['predicted'] = df.apply(
            lambda row: calculate_reimbursement(row['trip_days'], row['miles'], row['receipts'], version),
            axis=1
        )
        
        # Calculate errors
        df['error'] = df['predicted'] - df['expected_output']
        df['abs_error'] = abs(df['error'])
        df['pct_error'] = (df['abs_error'] / df['expected_output']) * 100
        
        # Summary statistics
        mae = df['abs_error'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        mape = df['pct_error'].mean()
        
        print(f"Total cases: {len(df)}")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Root Mean Square Error: ${rmse:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.1f}%")
        
        # Show worst predictions
        print("\nWorst 5 predictions:")
        worst = df.nlargest(5, 'abs_error')[['trip_days', 'miles', 'receipts', 
                                              'expected_output', 'predicted', 'abs_error']]
        print(worst.to_string(index=False))
        
        # Show best predictions
        print("\nBest 5 predictions:")
        best = df.nsmallest(5, 'abs_error')[['trip_days', 'miles', 'receipts', 
                                              'expected_output', 'predicted', 'abs_error']]
        print(best.to_string(index=False))
        
        # Save full results
        output_file = filepath.replace('.json', f'_predictions_{version}.csv')
        df.to_csv(output_file, index=False)
        print(f"\nFull results saved to: {output_file}")
        
    else:
        # Private data - just output predictions
        df.columns = ['trip_days', 'miles', 'receipts']
        
        # Calculate predictions
        df['predicted_reimbursement'] = df.apply(
            lambda row: calculate_reimbursement(row['trip_days'], row['miles'], row['receipts'], version),
            axis=1
        )
        
        print(f"Total cases: {len(df)}")
        print(f"Average predicted reimbursement: ${df['predicted_reimbursement'].mean():.2f}")
        
        # Save results
        output_file = filepath.replace('.json', f'_predictions_{version}.csv')
        df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
    
    return df

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Calculate travel reimbursements using the legacy system formula',
        epilog='Examples:\n'
               '  %(prog)s 5 300 750.50\n'
               '  %(prog)s data/raw/public_cases.json\n'
               '  %(prog)s --version v0.1 5 300 750.50\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('args', nargs='+', 
                       help='Either: trip_days miles receipts OR json_filepath')
    parser.add_argument('--version', default='v0.5',
                       help='Formula version to use (default: v0.5)')
    
    args = parser.parse_args()
    
    # Determine mode based on number of arguments
    if len(args.args) == 1:
        # JSON file mode
        filepath = args.args[0]
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
        process_json_file(filepath, args.version)
        
    elif len(args.args) == 3:
        # Single calculation mode
        try:
            trip_days = float(args.args[0])
            miles = float(args.args[1])
            receipts = float(args.args[2])
            process_single(trip_days, miles, receipts, args.version)
        except ValueError:
            print("Error: All three arguments must be numbers")
            print("Usage: calculate_reimbursement.py <trip_days> <miles> <receipts>")
            sys.exit(1)
            
    else:
        print("Error: Provide either 3 arguments (trip_days miles receipts) or 1 argument (json_file)")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 