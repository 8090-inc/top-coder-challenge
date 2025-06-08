"""Baseline model wrappers"""

import sys
sys.path.append('.')

from model_framework.core.base_model import BaseModel
from models.v5_practical_ensemble import calculate_reimbursement_v5
from models.cluster_models_optimized import calculate_reimbursement_v3


class V5Model(BaseModel):
    """Wrapper for v5 practical ensemble model"""
    
    def __init__(self):
        super().__init__(
            model_id="practical_ensemble",
            version="5.0",
            description="Practical ensemble with ML residual correction (baseline)"
        )
        
    def predict(self, trip_days, miles, receipts):
        return calculate_reimbursement_v5(trip_days, miles, receipts)


class V3Model(BaseModel):
    """Wrapper for v3 rule engine"""
    
    def __init__(self):
        super().__init__(
            model_id="rule_engine", 
            version="3.0",
            description="Optimized cluster-based rule engine"
        )
        
    def predict(self, trip_days, miles, receipts):
        return calculate_reimbursement_v3(trip_days, miles, receipts) 