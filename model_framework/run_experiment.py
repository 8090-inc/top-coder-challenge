"""Run model experiments"""

import sys
sys.path.append('.')

from model_framework.core.evaluator import ModelEvaluator
from model_framework.experiments.tracker import ExperimentTracker
from model_framework.models.baseline_models import V5Model, V3Model


def run_baseline_test():
    """Test the baseline models"""
    print("Model Testing Framework - Baseline Test")
    print("=" * 60)
    
    # Initialize components
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    # Test baseline models
    v5 = V5Model()
    v3 = V3Model()
    
    # Evaluate v5
    print("\n1. Evaluating V5 Baseline:")
    v5_metrics, v5_data = evaluator.evaluate(v5)
    tracker.log_experiment(v5_metrics, "Baseline v5 model")
    
    # Evaluate v3  
    print("\n2. Evaluating V3 Rule Engine:")
    v3_metrics, v3_data = evaluator.evaluate(v3)
    tracker.log_experiment(v3_metrics, "V3 rule engine only")
    
    # Compare them
    print("\n3. Comparing Models:")
    comparison, _, _ = evaluator.compare(v3, v5)
    
    # Show history
    print("\n4. Experiment History:")
    print(tracker.history[['model_name', 'mae', 'rmse', 'timestamp']].to_string())
    
    return v5_metrics, v3_metrics


def create_example_improvement():
    """Example of creating an improved model"""
    from model_framework.core.base_model import BaseModel
    from model_framework.models.baseline_models import V5Model
    
    class V5_1_ReceiptPenaltyFix(BaseModel):
        """Example: V5.1 with adjusted receipt penalties"""
        
        def __init__(self):
            super().__init__(
                model_id="practical_ensemble",
                version="5.1", 
                description="Testing adjusted .49 receipt penalty"
            )
            self.base_model = V5Model()
            
        def predict(self, trip_days, miles, receipts):
            # Get base prediction
            base_pred = self.base_model.predict(trip_days, miles, receipts)
            
            # Example: Adjust predictions for receipts ending in .49
            if int(receipts * 100) % 100 == 49:
                # Reduce penalty slightly
                adjustment = base_pred * 0.05  # 5% boost
                return round(base_pred + adjustment, 2)
            
            return base_pred
    
    # Test the improvement
    print("\n" + "=" * 60)
    print("Testing Example Improvement: V5.1")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    v5 = V5Model()
    v5_1 = V5_1_ReceiptPenaltyFix()
    
    # Compare
    comparison, data_v5, data_v5_1 = evaluator.compare(v5, v5_1)
    
    # Log experiment
    v5_1_metrics, _ = evaluator.evaluate(v5_1, verbose=False)
    tracker.log_experiment(v5_1_metrics, "Adjusted .49 receipt penalty")
    
    # Find cases that changed
    print("\nCases with .49 receipts that improved:")
    mask_49 = (data_v5['receipts'] * 100 % 100).astype(int) == 49
    improved = data_v5_1.loc[mask_49 & (data_v5_1['abs_error'] < data_v5['abs_error'])]
    
    if not improved.empty:
        print(improved[['trip_days', 'miles', 'receipts', 'expected_output', 'predicted', 'abs_error']].head())
    else:
        print("No improvements found for .49 cases")


if __name__ == "__main__":
    # Run baseline test
    run_baseline_test()
    
    # Show example improvement
    create_example_improvement() 