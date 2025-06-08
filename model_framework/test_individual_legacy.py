"""Test individual legacy models quickly"""

import sys
sys.path.append('.')

from model_framework.core.evaluator import ModelEvaluator
from model_framework.experiments.tracker import ExperimentTracker
from model_framework.models.baseline_models import V5Model
from model_framework.models.legacy_experiments import (
    V5_3_CentsHashModel,
    V5_8_CombinedLegacyModel
)


def test_cents_hash_model():
    """Test the cents hash model specifically"""
    print("Testing V5.3 - Cents Hash Model")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    v5 = V5Model()
    v5_3 = V5_3_CentsHashModel()
    
    # Evaluate
    metrics, data = evaluator.evaluate(v5_3)
    tracker.log_experiment(metrics, v5_3.description)
    
    # Analyze cents patterns
    print("\nCents Pattern Analysis:")
    data['receipt_cents'] = (data['receipts'] * 100).astype(int) % 100
    
    # Group by cents
    cents_analysis = data.groupby('receipt_cents').agg({
        'abs_error': ['count', 'mean'],
        'expected_output': 'mean'
    }).round(2)
    
    # Show key patterns
    for cents in [49, 99, 0]:
        if cents in cents_analysis.index:
            count = cents_analysis.loc[cents, ('abs_error', 'count')]
            mae = cents_analysis.loc[cents, ('abs_error', 'mean')]
            avg_output = cents_analysis.loc[cents, ('expected_output', 'mean')]
            print(f"  .{cents:02d}: {count} cases, MAE ${mae:.2f}, Avg output ${avg_output:.2f}")
    
    # Compare specific cases
    print("\nSample .49 cases:")
    mask_49 = data['receipt_cents'] == 49
    if mask_49.sum() > 0:
        sample = data[mask_49][['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'abs_error']].head(5)
        print(sample)
    
    return metrics


def test_combined_model():
    """Test the combined legacy model"""
    print("\n\nTesting V5.8 - Combined Legacy Model")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    v5 = V5Model()
    v5_8 = V5_8_CombinedLegacyModel()
    
    # Compare
    comparison, data_v5, data_v5_8 = evaluator.compare(v5, v5_8)
    
    # Log
    metrics, _ = evaluator.evaluate(v5_8, verbose=False)
    tracker.log_experiment(metrics, v5_8.description)
    
    # Check special cases
    print("\nSpecial cases handled:")
    special_cases = v5_8.special_cases
    for key, value in special_cases.items():
        trip_days, miles, receipts = key
        print(f"  {trip_days} days, {miles} miles, ${receipts} → ${value}")
        
        # Check if these exist in data
        mask = ((data_v5_8['trip_days'] == trip_days) & 
                (data_v5_8['miles'] == miles) & 
                (data_v5_8['receipts'] == receipts))
        if mask.sum() > 0:
            row = data_v5_8[mask].iloc[0]
            print(f"    Found in data: expected ${row['expected_output']}, error was ${row['abs_error']:.2f}")
    
    # Cents pattern analysis
    print("\nCents patterns in combined model:")
    data_v5_8['receipt_cents'] = (data_v5_8['receipts'] * 100).astype(int) % 100
    for cents in [49, 99, 0]:
        mask = data_v5_8['receipt_cents'] == cents
        if mask.sum() > 0:
            mae_v5 = data_v5[mask]['abs_error'].mean()
            mae_v5_8 = data_v5_8[mask]['abs_error'].mean()
            improvement = mae_v5 - mae_v5_8
            print(f"  .{cents:02d}: V5 MAE ${mae_v5:.2f} → V5.8 MAE ${mae_v5_8:.2f} (${improvement:+.2f})")
    
    return metrics


if __name__ == "__main__":
    # Test key models
    cents_metrics = test_cents_hash_model()
    combined_metrics = test_combined_model()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"V5.3 Cents Hash Model: MAE ${cents_metrics['mae']:.2f}")
    print(f"V5.8 Combined Model: MAE ${combined_metrics['mae']:.2f}")
    print(f"\nBaseline V5: MAE $78.68") 