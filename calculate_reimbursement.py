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
FORMULA_VERSION = "v0.4_refined"

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

def calculate_reimbursement(trip_days, miles, receipts, version="v0.4"):
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
    else:
        raise ValueError(f"Unknown formula version: {version}")

def process_single(trip_days, miles, receipts, version="v0.4"):
    """Process a single reimbursement calculation"""
    result = calculate_reimbursement(trip_days, miles, receipts, version)
    print(f"Formula Version: {FORMULA_VERSION}")
    print(f"Inputs: {trip_days} days, {miles} miles, ${receipts:.2f} receipts")
    print(f"Expected Reimbursement: ${result:.2f}")
    return result

def process_json_file(filepath, version="v0.4"):
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
    parser.add_argument('--version', default='v0.4',
                       help='Formula version to use (default: v0.4)')
    
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