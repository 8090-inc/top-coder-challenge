"""
Check case 995 which is a major outlier
"""

import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append('/Users/smortada/Documents/Personal/top-coder-challenge')
from models.cluster_router import assign_cluster_v2

# Load data
with open('/Users/smortada/Documents/Personal/top-coder-challenge/data/raw/public_cases.json', 'r') as f:
    data = json.load(f)

# Get case 995
case = data[995]
trip_days = case['input']['trip_duration_days']
miles = case['input']['miles_traveled']
receipts = case['input']['total_receipts_amount']
expected = case['expected_output']

print(f"Case 995 Analysis:")
print(f"  Trip days: {trip_days}")
print(f"  Miles: {miles}")
print(f"  Receipts: ${receipts:.2f}")
print(f"  Expected output: ${expected:.2f}")
print(f"  Assigned cluster: {assign_cluster_v2(trip_days, miles, receipts)}")

# Find similar cases
print("\nLooking for similar 1-day high-mile trips with low expected output...")

similar_cases = []
for i, c in enumerate(data):
    if (c['input']['trip_duration_days'] == 1 and 
        c['input']['miles_traveled'] > 900 and
        c['expected_output'] < 600):
        similar_cases.append({
            'case_id': i,
            'miles': c['input']['miles_traveled'],
            'receipts': c['input']['total_receipts_amount'],
            'expected': c['expected_output']
        })

if similar_cases:
    df_similar = pd.DataFrame(similar_cases)
    print(f"\nFound {len(df_similar)} similar cases:")
    print(df_similar.to_string())
else:
    print("\nNo similar cases found")

# Check if this is really a 1-day trip with >1000 miles
print(f"\nIs this realistic? 1 day with {miles} miles = {miles/24:.1f} mph average")
print("This might be a data error or special case (flight reimbursement?)")

# Look for pattern in receipt ending
cents = int(receipts * 100) % 100
print(f"\nReceipt cents: {cents}") 