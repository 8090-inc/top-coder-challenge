"""Template for creating new model experiments"""

import sys
sys.path.append('.')

from model_framework.core.base_model import BaseModel
from model_framework.core.evaluator import ModelEvaluator
from model_framework.experiments.tracker import ExperimentTracker
from model_framework.models.baseline_models import V5Model


# Template for a new model experiment
class MyImprovedModel(BaseModel):
    """
    Template for creating an improved model.
    
    Change this class name and customize the predict method.
    """
    
    def __init__(self):
        super().__init__(
            model_id="practical_ensemble",  # Keep same ID for version tracking
            version="5.2",  # Increment version
            description="Your improvement description here"
        )
        # Load the base model we're improving
        self.base_model = V5Model()
        
        # Add any custom parameters here
        self.my_parameter = 0.1
        
    def predict(self, trip_days, miles, receipts):
        """
        Your improved prediction logic here.
        
        Examples of improvements to try:
        1. Adjust penalties for specific receipt endings
        2. Add corrections for specific trip patterns
        3. Modify predictions for certain clusters
        4. Apply different logic for edge cases
        """
        # Get base prediction
        base_pred = self.base_model.predict(trip_days, miles, receipts)
        
        # ========== YOUR IMPROVEMENTS GO HERE ==========
        # Example: Boost short trips slightly
        if trip_days <= 2:
            base_pred *= 1.02  # 2% boost
        
        # Example: Adjust for high-mile single day trips
        if trip_days == 1 and miles > 800:
            base_pred += 50  # Add $50
        
        # ===============================================
        
        return round(base_pred, 2)


def test_improvement():
    """Test your improved model"""
    print("Testing Model Improvement")
    print("=" * 60)
    
    # Initialize
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    # Create models
    baseline = V5Model()
    improved = MyImprovedModel()
    
    # Compare
    print(f"\nComparing {baseline.name} vs {improved.name}")
    comparison, data_baseline, data_improved = evaluator.compare(baseline, improved)
    
    # Log the experiment
    metrics, _ = evaluator.evaluate(improved, verbose=False)
    tracker.log_experiment(metrics, improved.description)
    
    # Analyze specific cases that changed
    changed_mask = data_improved['predicted'] != data_baseline['predicted']
    changed_cases = data_improved[changed_mask]
    
    print(f"\nTotal cases changed: {changed_mask.sum()}")
    if not changed_cases.empty:
        print("\nSample of changed predictions:")
        cols = ['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'abs_error']
        print(changed_cases[cols].head(10))
        
        # Show improvement breakdown
        improved_mask = data_improved.loc[changed_mask, 'abs_error'] < data_baseline.loc[changed_mask, 'abs_error']
        print(f"\nOf changed cases:")
        print(f"  Improved: {improved_mask.sum()}")
        print(f"  Worsened: {(~improved_mask).sum()}")
    
    # Show current leaderboard
    print("\n" + "=" * 60)
    print("Model Leaderboard (sorted by MAE):")
    print("=" * 60)
    leaderboard = tracker.compare_to_baseline()
    if not leaderboard.empty:
        cols = ['model_name', 'mae', 'improvement_pct', 'description']
        print(leaderboard[cols].to_string(index=False))
    
    return improved


if __name__ == "__main__":
    # Test the improvement
    model = test_improvement()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Modify the MyImprovedModel class with your improvements")
    print("2. Run this script to test: python model_framework/create_experiment.py")
    print("3. Check results in model_framework/results/")
    print("4. Iterate and improve!") 