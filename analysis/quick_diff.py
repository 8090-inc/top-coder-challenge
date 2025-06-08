#!/usr/bin/env python3
"""
quick_diff.py: Run current scorer on all public cases, output errors sorted by abs(error) descending.
"""
import sys
sys.path.append('.')
import json
import pandas as pd
from src.final_calculator import calculate_reimbursement_final

with open('public_cases.json') as f:
    cases = json.load(f)

rows = []
for row in cases:
    inp = row['input']
    days = float(inp['trip_duration_days'])
    miles = float(inp['miles_traveled'])
    receipts = float(inp['total_receipts_amount'])
    expected = float(row['expected_output'])
    predicted = calculate_reimbursement_final(days, miles, receipts)
    error = predicted - expected
    rows.append({
        'days': days,
        'miles': miles,
        'receipts': receipts,
        'expected': expected,
        'predicted': predicted,
        'error': error,
        'abs_error': abs(error)
    })

df = pd.DataFrame(rows)
df = df.sort_values('abs_error', ascending=False)
df.to_csv('analysis/errors_blitz.csv', index=False)
print(df.head(100).to_string(index=False)) 