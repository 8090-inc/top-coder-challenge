"""Find improvement opportunities in model predictions"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from model_framework.core.evaluator import ModelEvaluator
from model_framework.models.baseline_models import V5Model


def find_high_error_patterns():
    """Find patterns in high-error cases"""
    print("Finding Improvement Opportunities")
    print("=" * 60)
    
    # Evaluate current model
    evaluator = ModelEvaluator()
    model = V5Model()
    metrics, data = evaluator.evaluate(model, verbose=False)
    
    print(f"Current MAE: ${metrics['mae']:.2f}")
    
    # Find high error cases
    high_error_threshold = 200
    high_errors = data[data['abs_error'] > high_error_threshold].copy()
    
    print(f"\nCases with error > ${high_error_threshold}: {len(high_errors)}")
    
    if len(high_errors) > 0:
        # Analyze patterns
        print("\nHigh Error Patterns:")
        print("-" * 40)
        
        # By trip length
        print("\nBy Trip Days:")
        trip_stats = high_errors.groupby('trip_days').agg({
            'abs_error': ['count', 'mean']
        }).round(2)
        print(trip_stats)
        
        # By receipt endings
        print("\nBy Receipt Endings:")
        high_errors['receipt_cents'] = (high_errors['receipts'] * 100).astype(int) % 100
        receipt_stats = high_errors.groupby('receipt_cents').agg({
            'abs_error': ['count', 'mean']
        }).round(2)
        print(receipt_stats.sort_values(('abs_error', 'count'), ascending=False).head(10))
        
        # Show worst cases
        print("\nWorst 10 Cases:")
        print("-" * 40)
        worst = high_errors.nlargest(10, 'abs_error')[
            ['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'abs_error']
        ]
        print(worst.to_string(index=False))
    
    # Find systematic biases
    print("\n" + "=" * 60)
    print("Systematic Biases:")
    print("-" * 40)
    
    # By trip days
    bias_by_days = data.groupby('trip_days').agg({
        'error': 'mean',
        'abs_error': 'mean',
        'trip_days': 'count'
    }).rename(columns={'trip_days': 'count'}).round(2)
    
    print("\nBias by Trip Days (avg error):")
    print(bias_by_days[bias_by_days['count'] >= 10].sort_values('error'))
    
    # Receipt pattern analysis
    print("\n" + "=" * 60)
    print("Receipt Pattern Analysis:")
    print("-" * 40)
    
    data['receipt_cents'] = (data['receipts'] * 100).astype(int) % 100
    cents_stats = data.groupby('receipt_cents').agg({
        'abs_error': ['count', 'mean'],
        'error': 'mean'
    }).round(2)
    
    # Find problematic cents patterns
    problematic_cents = cents_stats[cents_stats[('abs_error', 'mean')] > metrics['mae'] * 1.5]
    if not problematic_cents.empty:
        print("\nProblematic receipt endings (>1.5x average error):")
        print(problematic_cents.sort_values(('abs_error', 'mean'), ascending=False).head(10))
    
    # Cluster analysis
    print("\n" + "=" * 60) 
    print("Improvement Suggestions:")
    print("-" * 40)
    
    suggestions = []
    
    # Check for receipt ending issues
    if 49 in problematic_cents.index:
        mae_49 = problematic_cents.loc[49, ('abs_error', 'mean')]
        count_49 = problematic_cents.loc[49, ('abs_error', 'count')]
        suggestions.append(f"- Adjust .49 receipt penalty (affects {count_49} cases, avg error ${mae_49:.2f})")
    
    if 99 in problematic_cents.index:
        mae_99 = problematic_cents.loc[99, ('abs_error', 'mean')]
        count_99 = problematic_cents.loc[99, ('abs_error', 'count')]
        suggestions.append(f"- Adjust .99 receipt penalty (affects {count_99} cases, avg error ${mae_99:.2f})")
    
    # Check for systematic biases
    if any(bias_by_days['error'].abs() > 50):
        worst_bias_days = bias_by_days['error'].abs().idxmax()
        bias_value = bias_by_days.loc[worst_bias_days, 'error']
        suggestions.append(f"- Fix bias for {worst_bias_days}-day trips (avg error: ${bias_value:.2f})")
    
    # High error cases
    if len(high_errors) > 20:
        suggestions.append(f"- Create special handling for {len(high_errors)} high-error cases")
    
    for suggestion in suggestions:
        print(suggestion)
    
    return data


if __name__ == "__main__":
    data = find_high_error_patterns() 