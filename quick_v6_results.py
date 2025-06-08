"""Quick V6 results generation - minimal output"""

import sys
import os
sys.path.append('.')

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import json
import pandas as pd
from model_framework.models.v6_simplified import V6_SimplifiedModel

# Redirect stderr to suppress LightGBM warnings
import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def main():
    print("Starting V6 generation...")
    
    # Load training data
    train_df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Train model (suppressing output)
    print("Training model...")
    with suppress_output():
        model = V6_SimplifiedModel()
        model.train(train_df)
    
    # Load private cases
    with open('private_cases.json', 'r') as f:
        private_cases = json.load(f)
    
    # Generate predictions
    print("Generating predictions...")
    results = []
    
    for i, case in enumerate(private_cases):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(private_cases)} completed")
            
        trip_days = case['input']['trip_days']
        miles = case['input']['miles']
        receipts = case['input']['receipts']
        
        prediction = model.predict(trip_days, miles, receipts)
        results.append(prediction)
    
    # Save results
    output_file = 'v6_private_results.txt'
    with open(output_file, 'w') as f:
        f.write("expected_output\n")
        for result in results:
            f.write(f"{result}\n")
    
    print(f"\n✅ Results saved to {output_file}")
    print(f"Total predictions: {len(results)}")

if __name__ == "__main__":
    main() 