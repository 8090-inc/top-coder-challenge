#!/usr/bin/env python3
"""
Black Box Reimbursement Calculator - Version 2
Based on deeper analysis showing receipts dominate the calculation
"""
import sys
import math

def calculate_reimbursement(days, miles, receipts):
    """
    Revised calculation based on analysis showing:
    1. Receipts are the dominant factor (0.704 correlation)
    2. Miles/day has NEGATIVE correlation with output
    3. High receipts seem to cap or replace other components
    """
    
    # Start with base per diem
    total = days * 100.0
    
    # Mileage component - more conservative
    if miles <= 100:
        mileage = miles * 0.58
    elif miles <= 300:
        mileage = 100 * 0.58 + (miles - 100) * 0.45
    elif miles <= 600:
        mileage = 100 * 0.58 + 200 * 0.45 + (miles - 300) * 0.35
    else:
        mileage = 100 * 0.58 + 200 * 0.45 + 300 * 0.35 + (miles - 600) * 0.25
    
    # Receipt processing - the dominant factor
    if receipts == 0:
        receipt_component = 0
    elif receipts < 50:
        # Small receipts get moderate treatment, not huge multiplier
        receipt_component = receipts * 1.5
    elif receipts < 200:
        receipt_component = 50 * 1.5 + (receipts - 50) * 1.2
    elif receipts < 500:
        receipt_component = 50 * 1.5 + 150 * 1.2 + (receipts - 200) * 0.9
    elif receipts < 1000:
        receipt_component = 50 * 1.5 + 150 * 1.2 + 300 * 0.9 + (receipts - 500) * 0.7
    elif receipts < 1500:
        receipt_component = 50 * 1.5 + 150 * 1.2 + 300 * 0.9 + 500 * 0.7 + (receipts - 1000) * 0.5
    elif receipts < 2000:
        receipt_component = 50 * 1.5 + 150 * 1.2 + 300 * 0.9 + 500 * 0.7 + 500 * 0.5 + (receipts - 1500) * 0.3
    else:
        # High receipts get poor treatment
        receipt_component = 50 * 1.5 + 150 * 1.2 + 300 * 0.9 + 500 * 0.7 + 500 * 0.5 + 500 * 0.3 + (receipts - 2000) * 0.15
    
    # When receipts are very high, they seem to cap the total reimbursement
    if receipts > 1500:
        # High receipts reduce the impact of mileage
        mileage *= (2000 - min(receipts, 2000)) / 500
    
    # Efficiency penalty (not bonus!) for very high miles/day
    miles_per_day = miles / days
    if miles_per_day > 300:
        # Penalty for unrealistic travel
        efficiency_penalty = (miles_per_day - 300) * days * 0.5
        total -= efficiency_penalty
    
    # Add components
    total += mileage + receipt_component
    
    # Trip duration adjustments
    if days == 5:
        # Small bonus for 5-day trips
        total += 20
    
    # Long trips with high receipts get slightly better treatment
    if days >= 10 and receipts > 1000:
        total += min(50, (days - 10) * 5)
    
    # Cap single day trips with very high values
    if days == 1:
        if miles > 800:
            total = min(total, 1500)
        if receipts > 1500:
            total = min(total, 1400)
    
    return round(total, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: calculate_reimbursement.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = calculate_reimbursement(days, miles, receipts)
        print(f"{result:.2f}")
        
    except ValueError:
        print("Error: Invalid input. Please provide numeric values.")
        sys.exit(1)
