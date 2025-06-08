"""Test all legacy system experiments"""

import sys
sys.path.append('.')

from model_framework.core.evaluator import ModelEvaluator
from model_framework.experiments.tracker import ExperimentTracker
from model_framework.models.baseline_models import V5Model
from model_framework.models.legacy_experiments import (
    V5_2_TruncationModel,
    V5_3_CentsHashModel,
    V5_4_StepFunctionModel,
    V5_5_IntegerCentsModel,
    V5_6_FixedMultiplierModel,
    V5_7_SpecialCasesModel,
    V5_8_CombinedLegacyModel
)


def test_all_legacy_models():
    """Test all legacy experiments"""
    print("Testing Legacy System Quirks")
    print("=" * 60)
    
    # Initialize
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    # Create all models
    baseline = V5Model()
    models = [
        V5_2_TruncationModel(),
        V5_3_CentsHashModel(),
        V5_4_StepFunctionModel(),
        V5_5_IntegerCentsModel(),
        V5_6_FixedMultiplierModel(),
        V5_7_SpecialCasesModel(),
        V5_8_CombinedLegacyModel()
    ]
    
    # Test each model
    results = []
    for model in models:
        print(f"\n{'-' * 60}")
        print(f"Testing {model.name}: {model.description}")
        
        # Compare to baseline
        comparison, data_baseline, data_model = evaluator.compare(baseline, model)
        
        # Get full metrics
        metrics, _ = evaluator.evaluate(model, verbose=False)
        
        # Log experiment
        tracker.log_experiment(metrics, model.description)
        
        # Store results
        results.append({
            'model': model,
            'metrics': metrics,
            'comparison': comparison
        })
        
        # Show specific improvements
        print(f"\nKey metrics:")
        print(f"  Exact matches: {metrics['exact_matches']}")
        print(f"  Error < $50: {(data_model['abs_error'] < 50).sum()}")
        
        # Check cents patterns for relevant models
        if hasattr(model, '__class__') and model.__class__.__name__ in ['V5_3_CentsHashModel', 'V5_8_CombinedLegacyModel']:
            print("\nCents pattern performance:")
            data_model['receipt_cents'] = (data_model['receipts'] * 100).astype(int) % 100
            for cents in [49, 99, 0]:
                mask = data_model['receipt_cents'] == cents
                if mask.sum() > 0:
                    mae = data_model[mask]['abs_error'].mean()
                    print(f"  .{cents:02d} endings: MAE ${mae:.2f} ({mask.sum()} cases)")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'MAE':<10} {'vs V5':<15} {'Better':<8} {'Worse':<8}")
    print("-" * 70)
    
    for r in results:
        model_name = r['model'].name.replace('practical_ensemble_', '')
        mae = r['metrics']['mae']
        improvement = r['comparison']['mae_improvement']
        better = r['comparison']['better_cases']
        worse = r['comparison']['worse_cases']
        
        print(f"{model_name:<20} ${mae:<9.2f} ${improvement:>+8.2f} ({improvement/baseline.mae*100:+.1f}%) {better:<8} {worse:<8}")
    
    # Find best model
    best_result = min(results, key=lambda x: x['metrics']['mae'])
    best_model = best_result['model']
    print(f"\nBest model: {best_model.name} with MAE ${best_result['metrics']['mae']:.2f}")
    
    return results


def analyze_truncation_impact():
    """Deep dive into truncation vs rounding impact"""
    print("\n" + "=" * 60)
    print("TRUNCATION ANALYSIS")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    v5 = V5Model()
    v5_2 = V5_2_TruncationModel()
    
    _, data_v5, data_v5_2 = evaluator.compare(v5, v5_2, verbose=False)
    
    # Find cases where truncation made a difference
    diff_mask = data_v5['predicted'] != data_v5_2['predicted']
    diff_cases = data_v5_2[diff_mask]
    
    print(f"Cases affected by truncation: {diff_mask.sum()}")
    
    if diff_mask.sum() > 0:
        # Show impact
        print(f"Average change: ${(data_v5_2[diff_mask]['predicted'] - data_v5[diff_mask]['predicted']).mean():.2f}")
        
        # Check if truncation helps with exact matches
        v5_exact = (data_v5['abs_error'] < 0.01).sum()
        v5_2_exact = (data_v5_2['abs_error'] < 0.01).sum()
        print(f"Exact matches: V5={v5_exact}, V5.2={v5_2_exact}")
        
        # Show sample cases
        print("\nSample affected cases:")
        sample = diff_cases[['trip_days', 'miles', 'receipts', 'expected_output', 'predicted']].head(5)
        print(sample)


if __name__ == "__main__":
    # Test all models
    results = test_all_legacy_models()
    
    # Deep dive into truncation
    analyze_truncation_impact()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. The best performing model can be further refined")
    print("2. Combine successful patterns from different models")
    print("3. Test on private data to ensure generalization") 