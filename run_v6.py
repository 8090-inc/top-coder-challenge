#!/usr/bin/env python3
"""Run V6 model for a single prediction"""

import sys
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

# Add path for imports
sys.path.append('.')

import pandas as pd
from model_framework.models.v6_simplified import V6_SimplifiedModel

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_v6.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    # Parse arguments
    trip_days = int(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
    
    # Load training data and train model
    df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Train model (silently)
    import contextlib
    import io
    
    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.redirect_stdout(io.StringIO()):
            model = V6_SimplifiedModel()
            model.train(df)
    
    # Make prediction
    prediction = model.predict(trip_days, miles, receipts)
    
    # Output just the number (no formatting)
    print(f"{prediction:.2f}")

if __name__ == "__main__":
    main() 