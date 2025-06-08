"""Simple experiment tracking using CSV files"""

import pandas as pd
import os
from datetime import datetime
import json


class ExperimentTracker:
    """Track experiment results in CSV files"""
    
    def __init__(self, results_dir='model_framework/results'):
        self.results_dir = results_dir
        self.history_file = os.path.join(results_dir, 'experiment_history.csv')
        os.makedirs(results_dir, exist_ok=True)
        
        # Load or create history
        if os.path.exists(self.history_file):
            self.history = pd.read_csv(self.history_file)
        else:
            self.history = pd.DataFrame()
    
    def log_experiment(self, metrics, description=""):
        """Log experiment results"""
        # Add timestamp and description
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['description'] = description
        
        # Append to history
        self.history = pd.concat([self.history, pd.DataFrame([metrics])], ignore_index=True)
        
        # Save
        self.history.to_csv(self.history_file, index=False)
        
        # Also save individual experiment details
        exp_id = f"{metrics['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_file = os.path.join(self.results_dir, f"{exp_id}_metrics.json")
        
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'):  # numpy scalar
                metrics_json[k] = v.item()
            else:
                metrics_json[k] = v
                
        with open(exp_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        return exp_id
    
    def get_best_model(self, metric='mae'):
        """Find best model by metric"""
        if self.history.empty:
            return None
        return self.history.loc[self.history[metric].idxmin()]
    
    def get_model_history(self, model_id):
        """Get all experiments for a model"""
        if self.history.empty:
            return pd.DataFrame()
        return self.history[self.history['model_id'] == model_id].sort_values('timestamp')
    
    def compare_to_baseline(self, baseline_name='v5.0'):
        """Compare all models to baseline"""
        if self.history.empty:
            return pd.DataFrame()
        
        baseline = self.history[self.history['model_name'].str.contains(baseline_name)]
        if baseline.empty:
            return pd.DataFrame()
        
        baseline_mae = baseline['mae'].iloc[-1]  # Use most recent
        
        comparison = self.history.copy()
        comparison['mae_vs_baseline'] = comparison['mae'] - baseline_mae
        comparison['improvement_pct'] = (baseline_mae - comparison['mae']) / baseline_mae * 100
        
        return comparison.sort_values('mae') 