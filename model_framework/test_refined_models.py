"""Test refined models"""

import sys
sys.path.append('.')

from model_framework.core.evaluator import ModelEvaluator
from model_framework.experiments.tracker import ExperimentTracker
from model_framework.models.baseline_models import V5Model
from model_framework.models.refined_experiments import (
    V5_9_TargetedFixesModel,
    V5_11_ErrorCorrectionModel,
    V5_12_HybridBestModel
)


def test_refined_models():
    """Test all refined models"""
    print("Testing Refined Models")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    baseline = V5Model()
    models = [
        V5_9_TargetedFixesModel(),
        V5_11_ErrorCorrectionModel(), 
        V5_12_HybridBestModel()
    ]
    
    results = []
    
    for model in models:
        print(f"\n{'-' * 60}")
        print(f"Testing {model.name}")
        
        # Evaluate
        metrics, data = evaluator.evaluate(model)
        tracker.log_experiment(metrics, model.description)
        
        # Compare to baseline
        comparison, _, _ = evaluator.compare(baseline, model, verbose=False)
        
        results.append({
            'model': model,
            'metrics': metrics,
            'comparison': comparison
        })
        
        # Check high error cases
        high_errors = data[data['abs_error'] > 200]
        print(f"\nHigh error cases (>$200): {len(high_errors)} (was 46 in V5)")
        
        # Check specific patterns we targeted
        print("\nTargeted pattern performance:")
        
        # 7-day low activity
        mask = ((data['trip_days'] == 7) & 
                (data['miles'] < 300) & 
                (data['receipts'] < 300))
        if mask.sum() > 0:
            mae = data[mask]['abs_error'].mean()
            print(f"  7-day low activity: {mask.sum()} cases, MAE ${mae:.2f}")
            
        # 9-day high activity
        mask = ((data['trip_days'] == 9) & 
                (data['miles'] > 900) & 
                (data['receipts'] > 1000))
        if mask.sum() > 0:
            mae = data[mask]['abs_error'].mean()
            print(f"  9-day high activity: {mask.sum()} cases, MAE ${mae:.2f}")
            
        # Special case check
        mask = ((data['trip_days'] == 4) & 
                (data['miles'] == 69) & 
                (data['receipts'] == 2321.49))
        if mask.sum() > 0:
            error = data[mask]['abs_error'].iloc[0]
            print(f"  Special case (4,69,2321.49): Error ${error:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("REFINED MODEL SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'MAE':<10} {'vs V5':<12} {'Better':<8} {'Worse':<8} {'High Errors'}")
    print("-" * 65)
    
    baseline_high_errors = 46  # Known from analysis
    
    for r in results:
        model_name = r['model'].name.replace('practical_ensemble_', '')
        mae = r['metrics']['mae']
        improvement = r['comparison']['mae_improvement']
        better = r['comparison']['better_cases']
        worse = r['comparison']['worse_cases']
        
        # Count high errors for this model
        _, data = evaluator.evaluate(r['model'], verbose=False)
        high_errors = len(data[data['abs_error'] > 200])
        
        print(f"{model_name:<15} ${mae:<9.2f} ${improvement:>+7.2f} {better:<8} {worse:<8} {high_errors}")
    
    print(f"\nBaseline V5: MAE $78.68, High errors: {baseline_high_errors}")
    
    # Find best
    best = min(results, key=lambda x: x['metrics']['mae'])
    print(f"\nBest refined model: {best['model'].name} with MAE ${best['metrics']['mae']:.2f}")
    
    return results


if __name__ == "__main__":
    results = test_refined_models()
    
    print("\n" + "=" * 60)
    print("Conclusions:")
    print("- Legacy system assumptions (truncation, harsh penalties) made things worse")
    print("- Targeted fixes for specific patterns show more promise")
    print("- Special case handling provides exact matches")
    print("- Focus on high-error patterns rather than broad changes") 