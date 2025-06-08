#!/usr/bin/env python3
"""Quick evaluation of V5.13 Final Improved Model"""

import json
import time
import numpy as np
import sys
sys.path.append('.')

from model_framework.models.final_v5_improved import V5_Final_ImprovedModel


def evaluate_v513():
    """Evaluate V5.13 model"""
    print("🎯 Evaluating V5.13 Final Improved Model")
    print("=" * 60)
    
    # Load test cases
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Create model
    model = V5_Final_ImprovedModel()
    print(f"📊 Testing on {len(cases)} cases...")
    
    start_time = time.time()
    results = []
    
    for i, case in enumerate(cases):
        # Extract inputs
        trip_days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Calculate prediction
        predicted = model.predict(trip_days, miles, receipts)
        error = abs(predicted - expected)
        
        results.append({
            'expected': expected,
            'predicted': predicted,
            'error': error,
            'trip_days': trip_days,
            'miles': miles,
            'receipts': receipts
        })
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    errors = [r['error'] for r in results]
    mae = np.mean(errors)
    exact_matches = sum(1 for e in errors if e < 0.01)
    
    print(f"\n✅ Completed in {elapsed:.2f} seconds")
    print(f"\n📈 Results:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Exact matches: {exact_matches} ({exact_matches/len(cases)*100:.1f}%)")
    print(f"  Max error: ${max(errors):.2f}")
    print(f"  Speed: {len(cases)/elapsed:.0f} cases/second")
    
    # Show the exact matches
    if exact_matches > 0:
        print("\n🎯 Exact matches found:")
        for i, r in enumerate(results):
            if r['error'] < 0.01:
                print(f"  Case {i+1}: {r['trip_days']} days, {r['miles']} miles, ${r['receipts']:.2f}")
                print(f"    Output: ${r['predicted']:.2f}")
    
    # Compare to V5 baseline
    print(f"\n📊 Comparison:")
    print(f"  V5 Baseline MAE: $78.68")
    print(f"  V5.13 MAE: ${mae:.2f}")
    print(f"  Improvement: ${78.68 - mae:.2f}")
    
    return results


if __name__ == "__main__":
    evaluate_v513() 