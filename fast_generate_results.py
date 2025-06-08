#!/usr/bin/env python3
"""
Fast results generation for the Black Box Challenge
Uses batch processing with V5.13 model for maximum speed
"""

import json
import time
import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

# Import our best model (V5.13)
from model_framework.models.final_v5_improved import V5_Final_ImprovedModel

def load_private_cases():
    """Load private test cases from JSON file"""
    with open('private_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Convert to arrays for batch processing
    days = np.array([case['trip_duration_days'] for case in cases])
    miles = np.array([case['miles_traveled'] for case in cases])
    receipts = np.array([case['total_receipts_amount'] for case in cases])
    
    return days, miles, receipts, len(cases)

def generate_results():
    """Generate results for all private test cases"""
    print("🧾 Black Box Challenge - Fast Results Generation")
    print("=" * 50)
    print()
    
    # Load test cases
    print("Loading private test cases...")
    days, miles, receipts, num_cases = load_private_cases()
    print(f"✓ Loaded {num_cases} test cases")
    print()
    
    # Initialize model
    print("Initializing V5.13 model (our best)...")
    model = V5_Final_ImprovedModel()
    print("✓ Model initialized")
    print()
    
    # Process all cases in batch
    print(f"Processing {num_cases} cases in batch mode...")
    start_time = time.time()
    
    # Process cases efficiently
    predictions = []
    for i in range(num_cases):
        if i % 500 == 0 and i > 0:
            print(f"  Progress: {i}/{num_cases} cases processed...")
        pred = model.predict(days[i], miles[i], receipts[i])
        predictions.append(pred)
    
    predictions = np.array(predictions)
    elapsed = time.time() - start_time
    cases_per_sec = num_cases / elapsed
    print(f"✓ Processed {num_cases} cases in {elapsed:.2f} seconds ({cases_per_sec:.1f} cases/sec)")
    print()
    
    # Write results to file
    print("Writing results to private_results.txt...")
    with open('private_results.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred:.2f}\n")
    
    print("✓ Results written successfully")
    print()
    
    # Summary statistics
    print("📊 Summary Statistics:")
    print(f"  - Total cases: {num_cases}")
    print(f"  - Processing time: {elapsed:.2f} seconds")
    print(f"  - Speed: {cases_per_sec:.1f} cases/second")
    print(f"  - Average prediction: ${np.mean(predictions):.2f}")
    print(f"  - Min prediction: ${np.min(predictions):.2f}")
    print(f"  - Max prediction: ${np.max(predictions):.2f}")
    print()
    
    print("✅ Results generated successfully!")
    print("📄 Output saved to private_results.txt")
    print("📊 Each line contains the result for the corresponding test case")
    print()
    print("🎯 Next steps:")
    print("  1. Check private_results.txt - it contains one result per line")
    print("  2. Each line corresponds to the same-numbered test case in private_cases.json")
    print("  3. Submit your private_results.txt file when ready!")

if __name__ == "__main__":
    generate_results() 