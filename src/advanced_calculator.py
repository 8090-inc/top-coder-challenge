#!/usr/bin/env python3
"""
Advanced Legacy Reimbursement Calculator
Based on discovered non-linear patterns and thresholds

Key discoveries:
- Receipt threshold at $828 with diminishing returns above
- Mileage diminishing returns (6.15x difference)
- Trip duration categories with different per-day rates
"""

import math

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculate reimbursement using discovered non-linear patterns
    
    Args:
        trip_duration_days (int): Number of days traveling
        miles_traveled (int): Total miles traveled
        total_receipts_amount (float): Total receipt amount
        
    Returns:
        float: Calculated reimbursement amount (rounded to 2 decimal places)
    """
    days = trip_duration_days
    miles = miles_traveled
    receipts = total_receipts_amount
    
    # Component 1: Trip duration component (decreasing per-day rates)
    if days <= 3:
        # Short trips: higher per-day rate
        base_per_day = 200.0
        duration_component = days * base_per_day
    elif days <= 7:
        # Medium trips: moderate per-day rate
        base_per_day = 130.0
        duration_component = days * base_per_day
    else:
        # Long trips: lower per-day rate
        base_per_day = 85.0
        duration_component = days * base_per_day
    
    # Component 2: Mileage component with diminishing returns
    if miles <= 200:
        # High rate for low mileage
        mileage_component = miles * 0.85
    elif miles <= 500:
        # Medium rate for medium mileage
        mileage_component = 200 * 0.85 + (miles - 200) * 0.45
    else:
        # Low rate for high mileage
        mileage_component = 200 * 0.85 + 300 * 0.45 + (miles - 500) * 0.25
    
    # Component 3: Receipt component with threshold and diminishing returns
    receipt_threshold = 828.0
    
    if receipts <= receipt_threshold:
        # Below threshold: higher processing rate
        receipt_component = receipts * 0.85
    else:
        # Above threshold: diminishing returns
        base_receipt = receipt_threshold * 0.85
        excess_receipts = receipts - receipt_threshold
        
        # Apply exponential decay for excess receipts
        decay_factor = 0.0008  # Controls how quickly returns diminish
        excess_multiplier = math.exp(-decay_factor * excess_receipts)
        
        receipt_component = base_receipt + excess_receipts * (0.3 * excess_multiplier + 0.1)
    
    # Component 4: Interaction adjustments
    # Efficiency bonus for medium-efficiency trips
    efficiency = miles / days if days > 0 else 0
    
    efficiency_bonus = 0
    if 100 <= efficiency <= 300:
        # Sweet spot for efficiency
        efficiency_bonus = 20.0 * (days / 7.0)  # Scale with trip length
    
    # Small constant adjustment
    base_adjustment = 50.0
    
    # Final calculation
    total_reimbursement = (duration_component + 
                          mileage_component + 
                          receipt_component + 
                          efficiency_bonus + 
                          base_adjustment)
    
    return round(total_reimbursement, 2)

def calculate_reimbursement_v2(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Alternative implementation using decision tree insights
    """
    days = trip_duration_days
    miles = miles_traveled
    receipts = total_receipts_amount
    
    # Main decision tree logic based on discovered patterns
    
    if receipts <= 828.10:
        # Lower receipt path
        if days <= 4.5:
            if miles <= 583:
                base = 400 + days * 80 + miles * 0.6 + receipts * 0.9
            else:
                base = 450 + days * 85 + miles * 0.4 + receipts * 0.85
        else:
            if miles <= 624.5:
                base = 300 + days * 120 + miles * 0.5 + receipts * 0.8
            else:
                base = 350 + days * 125 + miles * 0.35 + receipts * 0.75
    else:
        # Higher receipt path (above $828)
        if days <= 5.5:
            if miles <= 621:
                base = 200 + days * 60 + miles * 0.3 + receipts * 0.4
            else:
                base = 250 + days * 65 + miles * 0.25 + receipts * 0.35
        else:
            if miles <= 644.5:
                base = 150 + days * 100 + miles * 0.4 + receipts * 0.45
            else:
                base = 200 + days * 105 + miles * 0.3 + receipts * 0.4
    
    return round(base, 2)

# Use the primary version as default
def calculate_reimbursement_optimized(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Optimized version combining insights from both approaches
    """
    # Try both methods and average them (ensemble approach)
    result1 = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
    result2 = calculate_reimbursement_v2(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Weighted average favoring the pattern-based approach
    final_result = 0.7 * result1 + 0.3 * result2
    
    return round(final_result, 2)

if __name__ == "__main__":
    # Test with some sample cases
    test_cases = [
        (3, 93, 1.42),  # Case from dataset
        (1, 55, 3.6),   # Case from dataset
        (5, 500, 1000), # Medium case
    ]
    
    for days, miles, receipts in test_cases:
        result1 = calculate_reimbursement(days, miles, receipts)
        result2 = calculate_reimbursement_v2(days, miles, receipts)
        result3 = calculate_reimbursement_optimized(days, miles, receipts)
        print(f"Case {days}d, {miles}mi, ${receipts}: v1=${result1}, v2=${result2}, optimized=${result3}") 