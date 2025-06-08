"""Quick evaluation of V6 model performance"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

from model_framework.models.v6_simplified import V6_SimplifiedModel
from models.v5_practical_ensemble import calculate_reimbursement_v5

def quick_eval():
    print("=== QUICK V6 EVALUATION ===\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Train V6
    print("Training V6 model...")
    import contextlib
    import io
    
    with contextlib.redirect_stderr(io.StringIO()):
        model = V6_SimplifiedModel()
        model.train(df)
    
    print("✓ Model trained\n")
    
    # Quick evaluation on training data
    print("Evaluating on training data (1000 cases)...")
    
    v5_errors = []
    v6_errors = []
    
    for idx, row in df.iterrows():
        actual = row['expected_output']
        
        # V5 prediction
        v5_pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
        v5_error = abs(v5_pred - actual)
        v5_errors.append(v5_error)
        
        # V6 prediction
        v6_pred = model.predict(row['trip_days'], row['miles'], row['receipts'])
        v6_error = abs(v6_pred - actual)
        v6_errors.append(v6_error)
    
    # Calculate metrics
    v5_mae = np.mean(v5_errors)
    v6_mae = np.mean(v6_errors)
    
    v5_max = np.max(v5_errors)
    v6_max = np.max(v6_errors)
    
    v5_median = np.median(v5_errors)
    v6_median = np.median(v6_errors)
    
    # Results
    print("\n=== RESULTS ===")
    print(f"{'Metric':<20} {'V5':>10} {'V6':>10} {'Improvement':>15}")
    print("-" * 60)
    print(f"{'MAE':<20} ${v5_mae:>9.2f} ${v6_mae:>9.2f} ${v5_mae - v6_mae:>14.2f}")
    print(f"{'Max Error':<20} ${v5_max:>9.2f} ${v6_max:>9.2f} ${v5_max - v6_max:>14.2f}")
    print(f"{'Median Error':<20} ${v5_median:>9.2f} ${v6_median:>9.2f} ${v5_median - v6_median:>14.2f}")
    
    improvement_pct = (v5_mae - v6_mae) / v5_mae * 100
    print(f"\n🎯 V6 Improvement: {improvement_pct:.1f}%")
    
    # Show error distribution
    print("\n=== ERROR DISTRIBUTION ===")
    for threshold in [10, 50, 100, 200]:
        v5_count = sum(1 for e in v5_errors if e < threshold)
        v6_count = sum(1 for e in v6_errors if e < threshold)
        print(f"Errors < ${threshold:<3}: V5={v5_count:>4} ({v5_count/len(df)*100:>5.1f}%), "
              f"V6={v6_count:>4} ({v6_count/len(df)*100:>5.1f}%)")
    
    # Show some examples
    print("\n=== SAMPLE PREDICTIONS ===")
    print("Showing cases with biggest improvement:")
    
    improvements = [v5_errors[i] - v6_errors[i] for i in range(len(df))]
    best_improvements = sorted(enumerate(improvements), key=lambda x: x[1], reverse=True)[:5]
    
    for idx, improvement in best_improvements:
        row = df.iloc[idx]
        v5_pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
        v6_pred = model.predict(row['trip_days'], row['miles'], row['receipts'])
        actual = row['expected_output']
        
        print(f"\nCase {idx}: Days={row['trip_days']}, Miles={row['miles']:.0f}, Receipts=${row['receipts']:.2f}")
        print(f"  Actual: ${actual:.2f}")
        print(f"  V5: ${v5_pred:.2f} (error ${abs(v5_pred - actual):.2f})")
        print(f"  V6: ${v6_pred:.2f} (error ${abs(v6_pred - actual):.2f})")
        print(f"  Improvement: ${improvement:.2f}")

if __name__ == "__main__":
    quick_eval() 