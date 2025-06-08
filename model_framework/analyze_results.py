"""Analyze experiment results"""

import sys
sys.path.append('.')

import pandas as pd
from model_framework.experiments.tracker import ExperimentTracker


def analyze_experiments():
    """Analyze all experiment results"""
    print("Experiment Analysis")
    print("=" * 60)
    
    tracker = ExperimentTracker()
    
    if tracker.history.empty:
        print("No experiments found. Run some experiments first!")
        return
    
    # Show all experiments sorted by MAE
    print("\nAll Experiments (sorted by MAE):")
    print("-" * 60)
    cols = ['model_name', 'mae', 'rmse', 'bias', 'exact_matches', 'runtime', 'description']
    available_cols = [c for c in cols if c in tracker.history.columns]
    print(tracker.history.sort_values('mae')[available_cols].to_string(index=False))
    
    # Show best model
    print("\n" + "=" * 60)
    best = tracker.get_best_model()
    print(f"Best Model: {best['model_name']} with MAE ${best['mae']:.2f}")
    
    # Compare to baseline
    print("\n" + "=" * 60)
    print("Comparison to V5 Baseline:")
    print("-" * 60)
    comparison = tracker.compare_to_baseline('v5.0')
    if not comparison.empty:
        cols = ['model_name', 'mae', 'improvement_pct']
        print(comparison[cols].to_string(index=False))
    
    # Show model evolution
    print("\n" + "=" * 60)
    print("Model Evolution (practical_ensemble versions):")
    print("-" * 60)
    ensemble_history = tracker.get_model_history('practical_ensemble')
    if not ensemble_history.empty:
        cols = ['version', 'mae', 'timestamp', 'description']
        available_cols = [c for c in cols if c in ensemble_history.columns]
        print(ensemble_history[available_cols].to_string(index=False))


if __name__ == "__main__":
    analyze_experiments() 