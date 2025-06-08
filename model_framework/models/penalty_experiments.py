"""Test legacy-style penalty calculations"""

import sys
import math
sys.path.append('.')

from model_framework.core.base_model import BaseModel
from models.v5_practical_ensemble import calculate_reimbursement_v5
from models.cluster_models_optimized import calculate_reimbursement_v3


class V5_14_LegacyPenaltyModel(BaseModel):
    """V5.14 - Legacy-style penalty calculation with truncation"""
    
    def __init__(self):
        super().__init__(
            model_id="practical_ensemble",
            version="5.14",
            description="Legacy-style penalty calculation (truncation + integer arithmetic)"
        )
        
    def apply_legacy_penalty(self, amount, receipts):
        """Apply penalty using legacy system logic"""
        cents = int(receipts * 100) % 100
        if cents == 49:
            # Use truncation instead of rounding
            return math.floor(amount * 0.341 * 100) / 100
        elif cents == 99:
            return math.floor(amount * 0.51 * 100) / 100
        return amount
        
    def predict(self, trip_days, miles, receipts):
        # Get v5 prediction WITHOUT penalty (we'll apply our own)
        # We need to get the pre-penalty amount from v3 rule engine
        v3_amount = calculate_reimbursement_v3(trip_days, miles, receipts)
        
        # Check if v3 already applied penalty
        cents = int(receipts * 100) % 100
        if cents in [49, 99]:
            # Reverse the penalty to get base amount
            if cents == 49:
                base_amount = v3_amount / 0.341
            else:  # cents == 99
                base_amount = v3_amount / 0.51
        else:
            base_amount = v3_amount
            
        # Now get v5's ML correction on the base
        v5_pred = calculate_reimbursement_v5(trip_days, miles, receipts)
        
        # Calculate v5's correction factor
        if v3_amount > 0:
            ml_factor = v5_pred / v3_amount
        else:
            ml_factor = 1.0
            
        # Apply ML correction to base
        corrected_amount = base_amount * ml_factor
        
        # Apply our legacy-style penalty
        final_amount = self.apply_legacy_penalty(corrected_amount, receipts)
        
        return round(final_amount, 2)


class V5_15_IntegerOnlyPenaltyModel(BaseModel):
    """V5.15 - Pure integer arithmetic for penalties"""
    
    def __init__(self):
        super().__init__(
            model_id="practical_ensemble",
            version="5.15",
            description="Pure integer cents arithmetic for penalty calculation"
        )
        
    def predict(self, trip_days, miles, receipts):
        # Get base v5 prediction
        v5_pred = calculate_reimbursement_v5(trip_days, miles, receipts)
        
        # Work entirely in integer cents
        receipts_cents = int(receipts * 100)
        amount_cents = int(v5_pred * 100)
        
        # Apply penalty in integer arithmetic
        cents_pattern = receipts_cents % 100
        
        if cents_pattern == 49:
            # 0.341 ≈ 341/1000
            amount_cents = (amount_cents * 341) // 1000
        elif cents_pattern == 99:
            # 0.51 = 51/100  
            amount_cents = (amount_cents * 51) // 100
            
        # Convert back to dollars
        return amount_cents / 100


class V5_16_ExactPenaltyFactorsModel(BaseModel):
    """V5.16 - Test if penalty factors should be exact fractions"""
    
    def __init__(self):
        super().__init__(
            model_id="practical_ensemble",
            version="5.16",
            description="Test exact fractional penalties (1/3, 1/2)"
        )
        
    def predict(self, trip_days, miles, receipts):
        # Get v5 prediction
        v5_pred = calculate_reimbursement_v5(trip_days, miles, receipts)
        
        # Check if penalties are already applied
        cents = int(receipts * 100) % 100
        
        if cents == 49:
            # Try 1/3 instead of 0.341
            # First reverse the 0.341 penalty
            base = v5_pred / 0.341
            # Apply 1/3 penalty
            return round(base / 3, 2)
        elif cents == 99:
            # Try 1/2 instead of 0.51
            # First reverse the 0.51 penalty
            base = v5_pred / 0.51
            # Apply 1/2 penalty  
            return round(base / 2, 2)
            
        return v5_pred


def test_penalty_experiments():
    """Test penalty calculation experiments"""
    from model_framework.core.evaluator import ModelEvaluator
    from model_framework.experiments.tracker import ExperimentTracker
    from model_framework.models.baseline_models import V5Model
    
    print("Testing Penalty Calculation Experiments")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    baseline = V5Model()
    models = [
        V5_14_LegacyPenaltyModel(),
        V5_15_IntegerOnlyPenaltyModel(),
        V5_16_ExactPenaltyFactorsModel()
    ]
    
    for model in models:
        print(f"\n{'-' * 60}")
        print(f"Testing {model.name}: {model.description}")
        
        # Evaluate
        metrics, data = evaluator.evaluate(model)
        tracker.log_experiment(metrics, model.description)
        
        # Check penalty cases specifically
        data['receipt_cents'] = (data['receipts'] * 100).astype(int) % 100
        
        print("\nPenalty case performance:")
        for cents in [49, 99]:
            mask = data['receipt_cents'] == cents
            if mask.sum() > 0:
                mae = data[mask]['abs_error'].mean()
                count = mask.sum()
                print(f"  .{cents} receipts: {count} cases, MAE ${mae:.2f}")
                
                # Show a sample
                sample = data[mask][['receipts', 'expected_output', 'predicted', 'abs_error']].head(3)
                print(f"    Sample:")
                for _, row in sample.iterrows():
                    print(f"      ${row['receipts']} → ${row['predicted']:.2f} (expected ${row['expected_output']:.2f}, error ${row['abs_error']:.2f})")
    
    print("\n" + "=" * 60)
    print("Summary: Testing different penalty calculation methods")


if __name__ == "__main__":
    test_penalty_experiments() 