#!/usr/bin/env python3
"""Fast evaluation script for testing different models"""

import json
import time
import numpy as np
import sys
sys.path.append('.')

from calculate_reimbursement import calculate_reimbursement
from model_framework.models.baseline_models import V5Model, V3Model
from model_framework.models.final_v5_improved import V5_Final_ImprovedModel


def evaluate_model(model, cases, model_name="Model"):
    """Evaluate a specific model on test cases"""
    print(f"\n🔍 Evaluating {model_name}...")
    start_time = time.time()
    
    results = []
    for i, case in enumerate(cases):
        if i % 200 == 0 and i > 0:
            print(f"  Progress: {i}/{len(cases)} cases...")
        
        # Extract inputs
        trip_days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Calculate prediction
        predicted = model.predict(trip_days, miles, receipts)
        error = abs(predicted - expected)
        
        results.append({
            'case_num': i + 1,
            'trip_days': trip_days,
            'miles': miles,
            'receipts': receipts,
            'expected': expected,
            'predicted': predicted,
            'error': error
        })
    
    # Calculate statistics
    errors_list = [r['error'] for r in results]
    exact_matches = sum(1 for e in errors_list if e < 0.01)
    avg_error = np.mean(errors_list)
    max_error = max(errors_list)
    
    elapsed = time.time() - start_time
    
    return {
        'name': model_name,
        'mae': avg_error,
        'exact_matches': exact_matches,
        'max_error': max_error,
        'elapsed': elapsed,
        'results': results
    }


def fast_compare_models():
    """Compare different models quickly"""
    print("🧾 Fast Model Comparison")
    print("=" * 60)
    
    # Load test cases once
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    print(f"📊 Loaded {len(cases)} test cases")
    
    # Test different models
    models_to_test = [
        (V3Model(), "V3 Rule Engine"),
        (V5Model(), "V5 Baseline"),
        (V5_Final_ImprovedModel(), "V5.13 Final Improved"),
    ]
    
    all_results = []
    
    for model, name in models_to_test:
        result = evaluate_model(model, cases, name)
        all_results.append(result)
    
    # Display comparison
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'MAE':<10} {'Exact':<10} {'Max Error':<12} {'Time'}")
    print("-" * 70)
    
    for r in all_results:
        exact_pct = r['exact_matches'] / len(cases) * 100
        print(f"{r['name']:<25} ${r['mae']:<9.2f} {r['exact_matches']:<4} ({exact_pct:>4.1f}%) ${r['max_error']:<11.2f} {r['elapsed']:.2f}s")
    
    # Find best model
    best = min(all_results, key=lambda x: x['mae'])
    print(f"\n🏆 Best Model: {best['name']} with MAE ${best['mae']:.2f}")
    
    # Show improvement over V3
    v3_result = next(r for r in all_results if "V3" in r['name'])
    for r in all_results:
        if r != v3_result:
            improvement = (v3_result['mae'] - r['mae']) / v3_result['mae'] * 100
            print(f"  {r['name']}: {improvement:.1f}% improvement over V3")
    
    # Analyze specific improvements in V5.13
    if len(all_results) >= 3:
        v5_result = all_results[1]
        v513_result = all_results[2]
        
        print(f"\n📈 V5.13 vs V5 Baseline Analysis:")
        
        # Find cases that improved
        improved_cases = []
        for i in range(len(cases)):
            v5_error = v5_result['results'][i]['error']
            v513_error = v513_result['results'][i]['error']
            if v513_error < v5_error:
                improved_cases.append({
                    'case': v513_result['results'][i],
                    'improvement': v5_error - v513_error
                })
        
        print(f"  Cases improved: {len(improved_cases)}")
        print(f"  Exact matches gained: {v513_result['exact_matches'] - v5_result['exact_matches']}")
        
        if improved_cases:
            print("\n  Top improvements:")
            improved_cases.sort(key=lambda x: x['improvement'], reverse=True)
            for imp in improved_cases[:5]:
                c = imp['case']
                print(f"    Case {c['case_num']}: ${imp['improvement']:.2f} improvement")
                print(f"      {c['trip_days']} days, {c['miles']} miles, ${c['receipts']:.2f}")
    
    return all_results


def quick_test_current():
    """Quick test of current implementation"""
    print("🚀 Quick Evaluation of Current Implementation")
    print("=" * 60)
    
    start = time.time()
    
    # Use the calculate_reimbursement.py method
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    errors = []
    exact = 0
    
    for case in cases:
        trip_days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculate_reimbursement(trip_days, miles, receipts)
        error = abs(predicted - expected)
        errors.append(error)
        
        if error < 0.01:
            exact += 1
    
    elapsed = time.time() - start
    mae = np.mean(errors)
    
    print(f"✅ Completed in {elapsed:.2f} seconds")
    print(f"📊 MAE: ${mae:.2f}")
    print(f"🎯 Exact matches: {exact}/{len(cases)} ({exact/len(cases)*100:.1f}%)")
    print(f"⚡ Speed: {len(cases)/elapsed:.0f} cases/second")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast model evaluation')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('--quick', action='store_true', help='Quick test current implementation')
    
    args = parser.parse_args()
    
    if args.compare:
        fast_compare_models()
    elif args.quick:
        quick_test_current()
    else:
        # Default: run both
        quick_test_current()
        print("\n" + "=" * 60)
        fast_compare_models() 