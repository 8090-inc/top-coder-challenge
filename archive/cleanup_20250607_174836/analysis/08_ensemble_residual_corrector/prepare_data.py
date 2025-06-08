#!/usr/bin/env python3
"""Prepare train and test data from public_cases.json"""

import json
import pandas as pd

# Load public cases
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
records = []
for case in data:
    record = {
        'trip_days': case['input']['trip_duration_days'],
        'miles_traveled': case['input']['miles_traveled'],
        'total_receipts_amount': case['input']['total_receipts_amount'],
        'expected_reimbursement': case['expected_output']
    }
    records.append(record)

df = pd.DataFrame(records)

# Save as train data (using all 1000 cases for training in this analysis)
df.to_csv('data/train.csv', index=False)

# For test data, we'll create a dummy test set with the same data
# In real scenario, this would be the private test set
df.drop('expected_reimbursement', axis=1).to_csv('data/test.csv', index=False)

print(f"Created data/train.csv with {len(df)} records")
print(f"Created data/test.csv with {len(df)} records") 