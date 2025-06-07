"""
Check penalty application for .49 and .99 receipts
"""

import sys
sys.path.append('/Users/smortada/Documents/Personal/top-coder-challenge')
from models.cluster_models import calculate_cluster_1a, apply_receipt_ending_penalty

# Test case 995 values
miles = 1082
receipts = 1809.49
expected = 446.94

# Calculate without penalty
base_amount = calculate_cluster_1a(1, miles, receipts)
print(f"Case 995 calculation:")
print(f"  Base amount (no penalty): ${base_amount:.2f}")

# Apply penalty
final_amount = apply_receipt_ending_penalty(base_amount, receipts)
print(f"  After .49 penalty (x0.341): ${final_amount:.2f}")
print(f"  Expected: ${expected:.2f}")
print(f"  Error: ${abs(final_amount - expected):.2f}")

# But wait - cluster 1a doesn't apply penalties in our current code!
print("\nWait... checking if penalty is actually applied for cluster 1a...")
print("In calculate_reimbursement_v2, we skip penalty for clusters 1a and 1b")
print("So the actual prediction would be: ${:.2f}".format(base_amount)) 