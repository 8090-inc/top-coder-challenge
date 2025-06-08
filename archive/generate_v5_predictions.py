import json
import pandas as pd
import numpy as np
from models.v5_practical_ensemble import PracticalEnsembleModel
import time

start_time = time.time()

# Load public cases - fastest JSON parsing
with open('data/raw/public_cases.json', 'r') as f:
    cases = json.load(f)

# Extract data from nested structure - using the column names expected by the model
data = []
for case in cases:
    data.append({
        'trip_days': case['input']['trip_duration_days'],
        'miles_traveled': case['input']['miles_traveled'],
        'total_receipts_amount': case['input']['total_receipts_amount'],
        'expected_output': case['expected_output']
    })

# Create DataFrame from extracted data
df = pd.DataFrame(data)

print(f"Data loaded: {len(df)} cases in {time.time() - start_time:.2f}s")

# Load model once
model_start = time.time()
model = PracticalEnsembleModel()
model.load('models/v5_practical_ensemble.pkl')
print(f"Model loaded in {time.time() - model_start:.2f}s")

# Predict in batch - already optimized in v5 model
predict_start = time.time()
predictions = model.predict_batch(df)
print(f"Predictions completed in {time.time() - predict_start:.2f}s")

# Vectorized operations for results
df['predicted'] = predictions
df['error'] = df['predicted'] - df['expected_output']
df['abs_error'] = np.abs(df['error'])

# Save with optimized settings
save_start = time.time()
# Rename columns for cents hash discovery compatibility
output_df = df.rename(columns={
    'miles_traveled': 'miles',
    'total_receipts_amount': 'receipts'
})
output_df.to_csv('public_cases_predictions_v5.csv', index=False)
print(f"CSV saved in {time.time() - save_start:.2f}s")

# Summary statistics
mae = df['abs_error'].mean()
max_error = df['abs_error'].max()
percentile_90 = df['abs_error'].quantile(0.9)

print(f"\n=== RESULTS ===")
print(f"Total cases: {len(df)}")
print(f"MAE: ${mae:.2f}")
print(f"Max error: ${max_error:.2f}")
print(f"90th percentile error: ${percentile_90:.2f}")
print(f"\nTotal execution time: {time.time() - start_time:.2f}s") 