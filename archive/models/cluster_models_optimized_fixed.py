"""
Optimized cluster-specific models with FIXED receipt penalty factors

Based on calibration analysis, the correct penalty factors are:
- .99 endings: 0.745 (not 0.51)
- .49 endings: 0.455 (not 0.341)
"""

# Import all the cluster calculation functions from the original
from models.cluster_models_optimized import (
    assign_cluster,
    calculate_cluster_0_optimized,
    calculate_cluster_0_low_mile_high_receipt_optimized,
    calculate_cluster_1a_optimized,
    calculate_cluster_1b_optimized,
    calculate_cluster_2_optimized,
    calculate_cluster_3_optimized,
    calculate_cluster_4_optimized,
    calculate_cluster_5_optimized,
    calculate_cluster_6_optimized
)

def apply_receipt_ending_penalty_fixed(amount, receipts):
    """Apply CORRECTED penalty for receipts ending in .49 or .99"""
    cents = int(receipts * 100) % 100
    
    if cents == 49:
        return amount * 0.455  # Corrected from 0.341
    elif cents == 99:
        return amount * 0.745  # Corrected from 0.51
    else:
        return amount


def calculate_reimbursement_v3_fixed(trip_days, miles, receipts):
    """
    Fixed v3 reimbursement calculation with corrected receipt penalties.
    
    Args:
        trip_days: Number of days for the trip
        miles: Miles traveled
        receipts: Total receipt amount
        
    Returns:
        Calculated reimbursement amount
    """
    # Import here to avoid circular dependency
    try:
        from models.cluster_router import assign_cluster_v2
    except ImportError:
        from cluster_router import assign_cluster_v2
    
    # Get cluster assignment
    cluster = assign_cluster_v2(trip_days, miles, receipts)
    
    # Route to appropriate calculation
    if cluster == '0':
        amount = calculate_cluster_0_optimized(trip_days, miles, receipts)
    elif cluster == '0_low_mile_high_receipt':
        amount = calculate_cluster_0_low_mile_high_receipt_optimized(trip_days, miles, receipts)
    elif cluster == '1a':
        amount = calculate_cluster_1a_optimized(trip_days, miles, receipts)
    elif cluster == '1b':
        amount = calculate_cluster_1b_optimized(trip_days, miles, receipts)
    elif cluster == '2':
        amount = calculate_cluster_2_optimized(trip_days, miles, receipts)
    elif cluster == '3':
        amount = calculate_cluster_3_optimized(trip_days, miles, receipts)
    elif cluster == '4':
        amount = calculate_cluster_4_optimized(trip_days, miles, receipts)
    elif cluster == '5':
        amount = calculate_cluster_5_optimized(trip_days, miles, receipts)
    elif cluster == '6':
        amount = calculate_cluster_6_optimized(trip_days, miles, receipts)
    else:
        # Fallback to cluster 0
        amount = calculate_cluster_0_optimized(trip_days, miles, receipts)
    
    # Apply FIXED receipt ending penalty to all clusters
    amount = apply_receipt_ending_penalty_fixed(amount, receipts)
    
    # Round to 2 decimal places
    return round(amount, 2) 