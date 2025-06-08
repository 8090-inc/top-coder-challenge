"""
Check the remaining outliers in v3
"""

import json
import sys
sys.path.append('/Users/smortada/Documents/Personal/top-coder-challenge')
from models.cluster_models_optimized import (
    calculate_cluster_0_optimized, 
    apply_receipt_ending_penalty
)

# Load data
with open('/Users/smortada/Documents/Personal/top-coder-challenge/data/raw/public_cases.json', 'r') as f:
    data = json.load(f)

# Check case 86 specifically
case_86 = data[86]
print("Case 86 Analysis:")
print(f"  Trip days: {case_86['input']['trip_duration_days']}")
print(f"  Miles: {case_86['input']['miles_traveled']}")  
print(f"  Receipts: ${case_86['input']['total_receipts_amount']:.2f}")
print(f"  Expected: ${case_86['expected_output']:.2f}")

# Calculate step by step
trip_days = case_86['input']['trip_duration_days']
miles = case_86['input']['miles_traveled']
receipts = case_86['input']['total_receipts_amount']

base = calculate_cluster_0_optimized(trip_days, miles, receipts)
print(f"\nBase calculation: ${base:.2f}")

final = apply_receipt_ending_penalty(base, receipts)
print(f"After penalty: ${final:.2f}")

cents = int(receipts * 100) % 100
print(f"\nReceipt cents: {cents}")
if cents == 49:
    print("Applied .49 penalty (x0.341)")

# Check the other high-receipt cases
print("\n" + "="*50)
print("Other high-receipt outliers:")

outlier_ids = [715, 504, 652, 783]
for case_id in outlier_ids:
    case = data[case_id]
    print(f"\nCase {case_id}:")
    print(f"  {case['input']['trip_duration_days']}d, "
          f"{case['input']['miles_traveled']}mi, "
          f"${case['input']['total_receipts_amount']:.2f}")
    print(f"  Expected: ${case['expected_output']:.2f}")
    
# Pattern analysis
print("\n" + "="*50)
print("PATTERN ANALYSIS:")
print("\nAll the high-receipt outliers (except case 86) have:")
print("- 8-9 days")
print("- High miles (500-1100)")
print("- Very high receipts ($2300-2500)")
print("- Expected outputs around $1500-1800")
print("\nBut our model predicts $2000-2300 for these.")
print("This suggests a cap or diminishing returns for very high receipts.") 