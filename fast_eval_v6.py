#!/usr/bin/env python3
"""Fast evaluation of V6 model with proper scoring"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

import contextlib
import io

from model_framework.models.v6_simplified import V6_SimplifiedModel
from models.v5_practical_ensemble import calculate_reimbursement_v5

def main():
    print("🧾 Black Box Challenge - V6 Model Fast Evaluation")
    print("=======================================================")
    print()
    
    # Load data
    print("📊 Loading and training V6 model...")
    df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Train V6 model once (silently)
    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.redirect_stdout(io.StringIO()):
            model = V6_SimplifiedModel()
            model.train(df)
    
    print("✓ Model trained!")
    print()
    print("📊 Running evaluation against 1,000 test cases...")
    print()
    
    # Evaluate all cases
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    total_error = 0.0
    max_error = 0.0
    max_error_case = ""
    errors = []
    results = []
    
    for idx, row in df.iterrows():
        trip_days = row['trip_days']
        miles = row['miles']
        receipts = row['receipts']
        expected = row['expected_output']
        
        # Get V6 prediction
        prediction = model.predict(trip_days, miles, receipts)
        error = abs(prediction - expected)
        
        successful_runs += 1
        errors.append(error)
        
        # Store result
        results.append({
            'case': idx + 1,
            'expected': expected,
            'predicted': prediction,
            'error': error,
            'trip_days': trip_days,
            'miles': miles,
            'receipts': receipts
        })
        
        # Check for exact match (within $0.01)
        if error < 0.01:
            exact_matches += 1
            
        # Check for close match (within $1.00)
        if error < 1.0:
            close_matches += 1
            
        # Update total error
        total_error += error
        
        # Track maximum error
        if error > max_error:
            max_error = error
            max_error_case = f"Case {idx+1}: {trip_days} days, {miles:.0f} miles, ${receipts:.2f} receipts"
    
    # Calculate results
    avg_error = total_error / successful_runs
    exact_pct = exact_matches * 100.0 / successful_runs
    close_pct = close_matches * 100.0 / successful_runs
    
    print("✅ Evaluation Complete!")
    print("")
    print("📈 Results Summary:")
    print(f"  Total test cases: {len(df)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print("")
    
    # Calculate score (same formula as eval.sh)
    score = avg_error * 100 + (len(df) - exact_matches) * 0.1
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print("")
    
    # Provide feedback based on exact matches
    if exact_matches == len(df):
        print("🏆 PERFECT SCORE! You have reverse-engineered the system completely!")
    elif exact_matches > 950:
        print("🥇 Excellent! You are very close to the perfect solution.")
    elif exact_matches > 800:
        print("🥈 Great work! You have captured most of the system behavior.")
    elif exact_matches > 500:
        print("🥉 Good progress! You understand some key patterns.")
    else:
        print("📚 Keep analyzing the patterns in the interviews and test cases.")
    
    print("")
    print("💡 Tips for improvement:")
    print("  Check these high-error cases:")
    
    # Sort by error and show top 5
    results_sorted = sorted(results, key=lambda x: x['error'], reverse=True)[:5]
    for result in results_sorted:
        print(f"    Case {result['case']}: {result['trip_days']} days, {result['miles']:.0f} miles, ${result['receipts']:.2f} receipts")
        print(f"      Expected: ${result['expected']:.2f}, Got: ${result['predicted']:.2f}, Error: ${result['error']:.2f}")
    
    # Compare with V5
    print("")
    print("📊 V6 vs V5 Comparison:")
    v5_errors = []
    for idx, row in df.iterrows():
        v5_pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
        v5_error = abs(v5_pred - row['expected_output'])
        v5_errors.append(v5_error)
    
    v5_mae = np.mean(v5_errors)
    v5_score = v5_mae * 100 + (len(df) - sum(1 for e in v5_errors if e < 0.01)) * 0.1
    
    print(f"  V5 MAE: ${v5_mae:.2f}, Score: {v5_score:.2f}")
    print(f"  V6 MAE: ${avg_error:.2f}, Score: {score:.2f}")
    print(f"  Improvement: ${v5_mae - avg_error:.2f} ({(v5_mae - avg_error) / v5_mae * 100:.1f}%)")

if __name__ == "__main__":
    main() 