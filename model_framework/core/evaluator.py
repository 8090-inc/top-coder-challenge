"""Model evaluation utilities"""

import pandas as pd
import numpy as np
import time


class ModelEvaluator:
    """Evaluate model performance on test data"""
    
    def __init__(self, test_data_path='public_cases_expected_outputs.csv'):
        """Load test data"""
        self.data = pd.read_csv(test_data_path)
        
    def evaluate(self, model, verbose=True):
        """Evaluate a model and return metrics"""
        if verbose:
            print(f"Evaluating {model.name}...")
        
        # Make predictions
        start_time = time.time()
        predictions = []
        
        for _, row in self.data.iterrows():
            pred = model.predict(row['trip_days'], row['miles'], row['receipts'])
            predictions.append(pred)
        
        runtime = time.time() - start_time
        
        # Calculate metrics
        self.data['predicted'] = predictions
        self.data['error'] = self.data['predicted'] - self.data['expected_output']
        self.data['abs_error'] = np.abs(self.data['error'])
        
        metrics = {
            'model_name': model.name,
            'model_id': model.model_id,
            'version': model.version,
            'mae': self.data['abs_error'].mean(),
            'rmse': np.sqrt((self.data['error'] ** 2).mean()),
            'bias': self.data['error'].mean(),
            'max_error': self.data['abs_error'].max(),
            'exact_matches': (self.data['abs_error'] < 0.01).sum(),
            'runtime': runtime,
            'cases_evaluated': len(self.data)
        }
        
        # Add error percentiles
        for pct in [50, 75, 90, 95]:
            metrics[f'error_p{pct}'] = np.percentile(self.data['abs_error'], pct)
        
        if verbose:
            print(f"  MAE: ${metrics['mae']:.2f}")
            print(f"  Runtime: {metrics['runtime']:.3f}s")
        
        return metrics, self.data.copy()
    
    def compare(self, model_a, model_b, verbose=True):
        """Compare two models"""
        metrics_a, data_a = self.evaluate(model_a, verbose)
        metrics_b, data_b = self.evaluate(model_b, verbose)
        
        # Calculate improvements
        comparison = {
            'model_a': model_a.name,
            'model_b': model_b.name,
            'mae_improvement': metrics_a['mae'] - metrics_b['mae'],
            'mae_improvement_pct': (metrics_a['mae'] - metrics_b['mae']) / metrics_a['mae'] * 100,
            'better_cases': (data_b['abs_error'] < data_a['abs_error']).sum(),
            'worse_cases': (data_b['abs_error'] > data_a['abs_error']).sum(),
            'unchanged_cases': (data_b['abs_error'] == data_a['abs_error']).sum(),
        }
        
        if verbose:
            print(f"\nComparison {model_a.name} vs {model_b.name}:")
            print(f"  MAE improvement: ${comparison['mae_improvement']:.2f} ({comparison['mae_improvement_pct']:.1f}%)")
            print(f"  Better: {comparison['better_cases']}, Worse: {comparison['worse_cases']}, Same: {comparison['unchanged_cases']}")
        
        return comparison, data_a, data_b 