import argparse
import math # For math.isclose for floating point comparisons

# --- PER DIEM ---
# v5: Calibrated per diem rates (Unchanged for v6, v7, v8, v9)
PER_DIEM_LOOKUP = {
    1: 187.29, 2: 274.07, 3: 263.59, 4: 323.35, 5: 395.12,
    6: 497.72, 7: 592.18, 8: 539.31, 9: 572.4, 10: 597.47,
    11: 656.82, 12: 681.46, 13: 781.79, 14: 789.09,
}
DEFAULT_PER_DIEM_RATE_FALLBACK = 50.0

# --- MILEAGE ---
# Tiered mileage rates (Unchanged from v3 onwards)
MILEAGE_RATE_TIER1 = 0.60
MILEAGE_TIER1_LIMIT = 50
MILEAGE_RATE_TIER2 = 0.50
MILEAGE_TIER2_LIMIT = 200 # Cumulative limit
MILEAGE_RATE_TIER3 = 0.35

# v7: Refined Optimal Efficiency Bonus scaling. Other efficiency adjustments unchanged. (Unchanged for v8, v9)
EFFICIENCY_LOW_THRESHOLD = 50
EFFICIENCY_OPTIMAL_LOWER = 150
EFFICIENCY_OPTIMAL_UPPER = 250
EFFICIENCY_HIGH_THRESHOLD = 300

# --- RECEIPTS ---
# v8: Low-Value Receipt Penalty cap is dynamic. High-tier general receipt rate reduced. (Unchanged for v9)
LOW_RECEIPT_THRESHOLD = 20.0
LOW_RECEIPT_PENALTY_FACTOR = -0.25
RECEIPT_SPECIAL_ENDING_FACTOR = 0.10
RECEIPT_TIER1_UPPER_BOUND = 600.0
RECEIPT_TIER1_RATE = 0.80
RECEIPT_TIER2_UPPER_BOUND = 1200.0
RECEIPT_TIER2_RATE = 0.50
RECEIPT_TIER3_RATE = 0.15

# --- MAX EFFORT BONUS (v9) ---
# v9: Added a bonus for trips meeting high thresholds for duration, miles, and receipts.
# This aims to address underestimation in the previous model's top error cases.
MAX_EFFORT_DAYS_THRESHOLD = 7
MAX_EFFORT_MILES_THRESHOLD = 700
MAX_EFFORT_RECEIPTS_THRESHOLD = 1000
MAX_EFFORT_BONUS_AMOUNT = 75.0


def calculate_per_diem(days: int) -> float:
    """(Unchanged from v5)"""
    return PER_DIEM_LOOKUP.get(days, days * DEFAULT_PER_DIEM_RATE_FALLBACK)

def calculate_mileage_reimbursement(miles: float, days: int) -> float:
    """(Unchanged from v7)"""
    tiered_reimbursement = 0.0
    if miles <= 0:
        tiered_reimbursement = 0.0
    elif miles <= MILEAGE_TIER1_LIMIT:
        tiered_reimbursement = miles * MILEAGE_RATE_TIER1
    elif miles <= MILEAGE_TIER2_LIMIT:
        tiered_reimbursement = (MILEAGE_TIER1_LIMIT * MILEAGE_RATE_TIER1) + \
                               ((miles - MILEAGE_TIER1_LIMIT) * MILEAGE_RATE_TIER2)
    else:
        tiered_reimbursement = (MILEAGE_TIER1_LIMIT * MILEAGE_RATE_TIER1) + \
                               ((MILEAGE_TIER2_LIMIT - MILEAGE_TIER1_LIMIT) * MILEAGE_RATE_TIER2) + \
                               ((miles - MILEAGE_TIER2_LIMIT) * MILEAGE_RATE_TIER3)

    efficiency_adjustment = 0.0
    if days > 0:
        efficiency = miles / days
        if efficiency < EFFICIENCY_LOW_THRESHOLD:
            efficiency_adjustment = max(-(30 + (days * 10)), -100.0)
        elif EFFICIENCY_OPTIMAL_LOWER <= efficiency <= EFFICIENCY_OPTIMAL_UPPER:
            efficiency_adjustment = min(30 + (days * 10), 150.0)
        elif efficiency > EFFICIENCY_HIGH_THRESHOLD :
            efficiency_adjustment = max(-(70 + (days * 10)), -200.0)
    elif miles > 0 and days <=0:
        efficiency_adjustment = max(-(30 + (1 * 10)), -100.0)
    return tiered_reimbursement + efficiency_adjustment

def calculate_receipt_contribution(receipts: float, per_diem_amount: float, days: int) -> float:
    """(Unchanged from v8)"""
    rounded_receipts = round(receipts, 2)
    fractional_part = round(rounded_receipts * 100) % 100
    is_ending_49 = math.isclose(fractional_part, 49)
    is_ending_99 = math.isclose(fractional_part, 99)

    if is_ending_49 or is_ending_99:
        return receipts * RECEIPT_SPECIAL_ENDING_FACTOR

    if receipts < LOW_RECEIPT_THRESHOLD:
        dynamic_cap = -(25 + (5 * days))
        calculated_penalty = per_diem_amount * LOW_RECEIPT_PENALTY_FACTOR
        return max(calculated_penalty, dynamic_cap)

    if receipts <= RECEIPT_TIER1_UPPER_BOUND:
        return receipts * RECEIPT_TIER1_RATE
    elif receipts <= RECEIPT_TIER2_UPPER_BOUND:
        return (RECEIPT_TIER1_UPPER_BOUND * RECEIPT_TIER1_RATE) + \
               ((receipts - RECEIPT_TIER1_UPPER_BOUND) * RECEIPT_TIER2_RATE)
    else:
        return (RECEIPT_TIER1_UPPER_BOUND * RECEIPT_TIER1_RATE) + \
               ((RECEIPT_TIER2_UPPER_BOUND - RECEIPT_TIER1_UPPER_BOUND) * RECEIPT_TIER2_RATE) + \
               ((receipts - RECEIPT_TIER2_UPPER_BOUND) * RECEIPT_TIER3_RATE)

def calculate_total_reimbursement_with_components(days: int, miles: float, receipts: float) -> tuple[float, float, float, float, float]:
    """
    Calculates total reimbursement and also returns its individual components.
    v9: Adds a 'Max Effort Trip' bonus component.
    Returns: (total_reimbursement, per_diem_amount, mileage_amount, receipts_amount, max_effort_bonus)
    """
    per_diem_amount = calculate_per_diem(days)
    mileage_amount = calculate_mileage_reimbursement(miles, days)
    receipts_amount = calculate_receipt_contribution(receipts, per_diem_amount, days)

    sub_total = per_diem_amount + mileage_amount + receipts_amount

    max_effort_bonus = 0.0
    # v9: Apply 'Max Effort Trip' bonus
    if days >= MAX_EFFORT_DAYS_THRESHOLD and \
       miles >= MAX_EFFORT_MILES_THRESHOLD and \
       receipts >= MAX_EFFORT_RECEIPTS_THRESHOLD:
        max_effort_bonus = MAX_EFFORT_BONUS_AMOUNT

    total_reimbursement = sub_total + max_effort_bonus

    return total_reimbursement, per_diem_amount, mileage_amount, receipts_amount, max_effort_bonus


def main():
    parser = argparse.ArgumentParser(description="Calculate travel reimbursement based on refined hypotheses (v9 - Max Effort Bonus).")
    parser.add_argument("--days", type=int, required=True, help="Total number of trip days")
    parser.add_argument("--miles", type=float, required=True, help="Total miles traveled")
    parser.add_argument("--receipts", type=float, required=True, help="Total amount from receipts")
    parser.add_argument("--components", action="store_true", help="Output individual components instead of just total")

    args = parser.parse_args()

    total_reimbursement, per_diem, mileage, receipt_comp, effort_bonus = calculate_total_reimbursement_with_components(args.days, args.miles, args.receipts)

    if args.components:
        # v9: Added MaxEffortBonus to component output
        print(f"PerDiem:{per_diem:.2f},Mileage:{mileage:.2f},Receipt:{receipt_comp:.2f},MaxEffortBonus:{effort_bonus:.2f},Total:{total_reimbursement:.2f}")
    else:
        print(f"{total_reimbursement:.2f}")

if __name__ == "__main__":
    main()
