#!/usr/bin/env python3
"""
Legacy Reimbursement Calculator
Discovered Formula: Linear combination approach

Formula: 266.71 + 50.05*days + 0.446*miles + 0.383*receipts
"""

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculate reimbursement using discovered linear formula
    
    Args:
        trip_duration_days (int): Number of days traveling
        miles_traveled (int): Total miles traveled
        total_receipts_amount (float): Total receipt amount
        
    Returns:
        float: Calculated reimbursement amount (rounded to 2 decimal places)
    """
    # Linear formula discovered through regression analysis
    base_constant = 266.71
    days_coefficient = 50.05
    miles_coefficient = 0.446
    receipts_coefficient = 0.383
    
    reimbursement = (base_constant + 
                    days_coefficient * trip_duration_days +
                    miles_coefficient * miles_traveled +
                    receipts_coefficient * total_receipts_amount)
    
    return round(reimbursement, 2)

if __name__ == "__main__":
    # Test with a sample case
    test_reimbursement = calculate_reimbursement(5, 500, 1000)
    print(f"Test calculation: {test_reimbursement}") 