"""Generate V6 model results for submission"""

import sys
sys.path.append('.')

import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
from model_framework.models.v6_simplified import V6_SimplifiedModel

def generate_results():
    """Generate results using V6 model"""
    
    print("Loading training data...")
    train_df = pd.read_csv('public_cases_expected_outputs.csv')
    
    print("Training V6 model...")
    model = V6_SimplifiedModel()
    model.train(train_df)
    
    print("\nGenerating predictions for private cases...")
    
    # Load private cases - it's a list directly
    with open('private_cases.json', 'r') as f:
        private_cases = json.load(f)
    
    results = []
    
    for i, case in enumerate(private_cases):
        if i % 1000 == 0:
            print(f"  Processing case {i}/{len(private_cases)}...")
            
        # Private cases use 'input' field
        trip_days = case['input']['trip_days']
        miles = case['input']['miles']
        receipts = case['input']['receipts']
        
        # Get prediction
        prediction = model.predict(trip_days, miles, receipts)
        
        results.append({
            'trip_days': trip_days,
            'miles': miles,
            'receipts': receipts,
            'expected_output': prediction
        })
    
    # Save results
    output_file = 'v6_private_results.txt'
    with open(output_file, 'w') as f:
        f.write("expected_output\n")
        for result in results:
            f.write(f"{result['expected_output']}\n")
    
    print(f"\nResults saved to {output_file}")
    print(f"Total predictions: {len(results)}")
    
    # Show sample results
    print("\nSample predictions:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"  Case {i}: Days={r['trip_days']}, Miles={r['miles']:.0f}, Receipts=${r['receipts']:.2f} -> ${r['expected_output']:.2f}")
    
    return results

if __name__ == "__main__":
    results = generate_results() 