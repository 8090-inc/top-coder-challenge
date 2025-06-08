#!/usr/bin/env python3
"""
Final Legacy Reimbursement Calculator (GBDT version)
Combines bucketed GBDT with special handling for extreme cases

Key insight: Very high receipts (>$1800) seem to have severe caps
that the model doesn't capture well.
"""

import sys
sys.path.append('src')
from calculate_reimbursement_gbdt_short import calculate_reimbursement_gbdt_short
from calculate_reimbursement_gbdt_mid import calculate_reimbursement_gbdt_mid
from calculate_reimbursement_gbdt_long import calculate_reimbursement_gbdt_long
from micro_rules import apply_micro_rules

def bucket_of(days: float) -> str:
    if days <= 4:
        return "short"
    elif days <= 8:
        return "mid"
    else:
        return "long"

def calculate_reimbursement_gbdt(days, miles, receipts):
    bucket = bucket_of(days)
    if bucket == "short":
        result = calculate_reimbursement_gbdt_short(trip_duration_days=days, miles_traveled=miles, total_receipts_amount=receipts)
    elif bucket == "mid":
        result = calculate_reimbursement_gbdt_mid(trip_duration_days=days, miles_traveled=miles, total_receipts_amount=receipts)
    else:
        result = calculate_reimbursement_gbdt_long(trip_duration_days=days, miles_traveled=miles, total_receipts_amount=receipts)
    if result is None:
        print(f"[WARN] GBDT scorer returned None for days={days}, miles={miles}, receipts={receipts}, bucket={bucket}")
        return 50.0
    return result

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Final calculator with extreme case handling (GBDT base)
    """
    days = trip_duration_days
    miles = miles_traveled
    receipts = total_receipts_amount

    # Get base prediction from GBDT
    base_prediction = calculate_reimbursement_gbdt(days, miles, receipts)

    # Apply extreme case adjustments
    if receipts > 1800:
        penalty_factor = min(0.6, 1800 / receipts)  # Cap at 60% of model prediction
        base_prediction *= penalty_factor
        if receipts > 2000:
            base_prediction -= 100
    elif receipts > 1400:
        penalty_factor = 0.8
        base_prediction *= penalty_factor
    efficiency = miles / days if days > 0 else 0
    if efficiency > 800:
        extreme_penalty = (efficiency - 800) * 0.5
        base_prediction -= extreme_penalty
    if receipts < 10:
        low_receipt_bonus = (10 - receipts) * 5
        base_prediction += low_receipt_bonus
    if base_prediction < 50:
        base_prediction = 50
    return round(base_prediction, 2)

def calculate_reimbursement_conservative(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    More conservative approach for extreme cases (GBDT base)
    """
    days = trip_duration_days
    miles = miles_traveled
    receipts = total_receipts_amount
    base_prediction = calculate_reimbursement_gbdt(days, miles, receipts)
    if receipts > 2000:
        max_allowed = 200 + days * 50 + min(miles * 0.2, 200) + min(receipts * 0.1, 200)
        base_prediction = min(base_prediction, max_allowed)
    elif receipts > 1500:
        max_allowed = 300 + days * 60 + min(miles * 0.3, 300) + min(receipts * 0.2, 300)
        base_prediction = min(base_prediction, max_allowed)
    return round(base_prediction, 2)

def calculate_reimbursement_final(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Final ensemble approach (GBDT base)
    """
    base_pred = calculate_reimbursement_gbdt(trip_duration_days, miles_traveled, total_receipts_amount)
    patched = apply_micro_rules(trip_duration_days, miles_traveled, total_receipts_amount, base_pred)
    return round(patched, 2)

if __name__ == "__main__":
    # Test with the worst cases
    worst_cases = [
        (4, 69, 2321.49, 322.00),      # Case 152
        (1, 1082, 1809.49, 446.94),   # Case 996  
        (8, 795, 1645.99, 644.69),    # Case 684
        (5, 516, 1878.49, 669.85),    # Case 711
        (3, 93, 1.42, 364.51),        # Normal case for comparison
    ]
    for days, miles, receipts, expected in worst_cases:
        gbdt_pred = calculate_reimbursement_gbdt(days, miles, receipts)
        final_pred = calculate_reimbursement_final(days, miles, receipts)
        gbdt_error = abs(gbdt_pred - expected)
        final_error = abs(final_pred - expected)
        improvement = "✅" if final_error < gbdt_error else "❌"
        print(f"Case {days}d, {miles}mi, ${receipts}:")
        print(f"  Expected: ${expected}")
        print(f"  GBDT: ${gbdt_pred} (error: ${gbdt_error:.2f})")
        print(f"  Final: ${final_pred} (error: ${final_error:.2f}) {improvement}")
        print() 