"""Base model class for reimbursement models"""

from abc import ABC, abstractmethod
from datetime import datetime


class BaseModel(ABC):
    """Abstract base class for all reimbursement models"""
    
    def __init__(self, model_id, version, description):
        self.model_id = model_id
        self.version = version
        self.description = description
        self.created_at = datetime.now()
        
    @abstractmethod
    def predict(self, trip_days, miles, receipts):
        """Make a single prediction"""
        pass
    
    @property
    def name(self):
        """Model name for display"""
        return f"{self.model_id}_v{self.version}"
    
    def __repr__(self):
        return f"<Model {self.name}: {self.description}>" 