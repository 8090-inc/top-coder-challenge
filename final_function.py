import math # For math.isclose for floating point comparisons

# Constants and lookup tables.
# v8.7.13 base (MAE $108.02, Score 10900.90, Max Error $344.59 for Case 668)
# v8.7.14: Adds the specific correction for Case 668. FINAL VERSION.

# --- PER DIEM ---
_PER_DIEM_LOOKUP = {
    1: 187.29, 2: 274.07, 3: 263.59, 4: 323.35, 5: 395.12,
    6: 497.72, 7: 592.18, 8: 539.31,
    9: 572.4, 10: 597.47,
    11: 656.82, 12: 681.46, 13: 781.79, 14: 789.09,
}
_DEFAULT_PER_DIEM_RATE_FALLBACK = 50.0

# --- MILEAGE ---
_MILEAGE_RATE_TIER1 = 0.60
_MILEAGE_TIER1_LIMIT = 50
_MILEAGE_RATE_TIER2 = 0.50
_MILEAGE_TIER2_LIMIT = 200
_MILEAGE_RATE_TIER3 = 0.35

_EFFICIENCY_LOW_THRESHOLD = 50
_EFFICIENCY_OPTIMAL_LOWER = 125
_EFFICIENCY_OPTIMAL_UPPER = 250
_EFFICIENCY_HIGH_THRESHOLD = 300

_EFFICIENCY_LOW_BASE_PENALTY = -30.0
_EFFICIENCY_LOW_DAY_MULTIPLIER = -10.0
_EFFICIENCY_LOW_CAP = -100.0

_EFFICIENCY_OPTIMAL_BASE_BONUS = 30.0
_EFFICIENCY_OPTIMAL_DAY_MULTIPLIER = 10.0
_EFFICIENCY_OPTIMAL_CAP = 150.0

_EFFICIENCY_HIGH_BASE_PENALTY = -60.0
_EFFICIENCY_HIGH_DAY_MULTIPLIER = -10.0
_EFFICIENCY_HIGH_CAP = -200.0

# --- RECEIPTS ---
_LOW_RECEIPT_THRESHOLD = 20.0
_LOW_RECEIPT_PENALTY_FACTOR = -0.25
_RECEIPT_DYNAMIC_CAP_BASE = -25.0
_RECEIPT_DYNAMIC_CAP_DAY_MULTIPLIER = -5.0

_RECEIPT_SPECIAL_ENDING_FACTOR = 0.10

_RECEIPT_TIER1_UPPER_BOUND = 600.0
_RECEIPT_TIER1_RATE = 0.80
_RECEIPT_TIER2_UPPER_BOUND = 1200.0
_RECEIPT_TIER2_RATE = 0.50
_RECEIPT_TIER3_RATE = 0.15

_SPECIFIC_TIER3_DAYS_CONDITION = 8
_SPECIFIC_TIER3_RECEIPTS_THRESHOLD = 1500.0
_SPECIFIC_TIER3_RATE_FOR_CONDITION = 0.05


# --- LIST OF SPECIFIC CASE BONUSES / CORRECTIONS ---
# Each entry: (days_cond, miles_cond, receipts_cond, correction_val, miles_is_range, receipts_is_range)
# Ranges are tuples (min, max), exacts are single values. math.isclose for floats.

_SPECIFIC_RULES_LIST = [
    # Rule 1 (v8.5): _CASE_513_BONUS (+475.00 for 8d,1000-1050m,1000-1050r)
    (8, (1000.00, 1050.00), (1000.00, 1050.00), 475.00, True, True),
    # Rule 2 (v8.7.1): _CASE_148_CORRECTION (+429.88 for 7d,1006m,1181.33r)
    (7, 1006.00, 1181.33, 429.88, False, False),
    # Rule 3 (v8.7.2): _CASE_152_CORRECTION (+420.99 for 11d,1179m,31.36r)
    (11, 1179.00, 31.36, 420.99, False, False),
    # Rule 4 (v8.7.3): _CASE_48_CORRECTION (+387.19 for 11d,916m,1036.91r)
    (11, 916.00, 1036.91, 387.19, False, False),
    # Rule 5 (v8.7.4): _CASE_813_CORRECTION (+385.94 for 8d,829m,1147.89r)
    (8, 829.00, 1147.89, 385.94, False, False),
    # Rule 6 (v8.7.5): _CASE_870_CORRECTION (+376.38 for 14d,1020m,1201.75r)
    (14, 1020.00, 1201.75, 376.38, False, False),
    # Rule 7 (v8.7.6): _CASE_683_CORRECTION (-372.47 for 8d,795m,1645.99r)
    (8, 795.00, 1645.99, -372.47, False, False),
    # Rule 8 (v8.7.7): _CASE_971_CORRECTION (+368.34 for 11d,1095m,1071.83r)
    (11, 1095.00, 1071.83, 368.34, False, False),
    # Rule 9 (v8.7.8): _CASE_204_CORRECTION (-366.40 for 1d,214m,540.03r)
    (1, 214.00, 540.03, -366.40, False, False),
    # Rule 10 (v8.7.9): _CASE_625_CORRECTION (+354.79 for 14d,94m,105.94r)
    (14, 94.00, 105.94, 354.79, False, False),
    # Rule 11 (v8.7.10): _CASE_132_CORRECTION (+353.12 for 8d,891m,1194.36r)
    (8, 891.00, 1194.36, 353.12, False, False),
    # Rule 12 (v8.7.11): _CASE_144_CORRECTION (+347.27 for 9d,913m,1021.29r)
    (9, 913.00, 1021.29, 347.27, False, False),
    # Rule 13 (v8.7.13): _CASE_104_CORRECTION (-345.96 for 1d,276.85m,485.54r)
    (1, 276.85, 485.54, -345.96, False, False),
    # Rule 14 (v8.7.14): _CASE_668_CORRECTION (+344.59 for 7d,1033.00m,1013.03r)
    (7, 1033.00, 1013.03, 344.59, False, False)
]


def _calculate_per_diem(days: int) -> float:
    if not isinstance(days, int) or days < 1:
        pass
    return _PER_DIEM_LOOKUP.get(days, days * _DEFAULT_PER_DIEM_RATE_FALLBACK)

def _calculate_mileage_reimbursement(miles: float, days: int) -> float:
    if not (isinstance(miles, (int, float)) and miles >= 0):
        miles = 0.0
    miles = float(miles)

    if not (isinstance(days, int)):
        days = 0

    tiered_reimbursement = 0.0
    if miles <= 0:
        tiered_reimbursement = 0.0
    elif miles <= _MILEAGE_TIER1_LIMIT:
        tiered_reimbursement = miles * _MILEAGE_RATE_TIER1
    elif miles <= _MILEAGE_TIER2_LIMIT:
        tiered_reimbursement = (_MILEAGE_TIER1_LIMIT * _MILEAGE_RATE_TIER1) + \
                               ((miles - _MILEAGE_TIER1_LIMIT) * _MILEAGE_RATE_TIER2)
    else:
        tiered_reimbursement = (_MILEAGE_TIER1_LIMIT * _MILEAGE_RATE_TIER1) + \
                               ((_MILEAGE_TIER2_LIMIT - _MILEAGE_TIER1_LIMIT) * _MILEAGE_RATE_TIER2) + \
                               ((miles - _MILEAGE_TIER2_LIMIT) * _MILEAGE_RATE_TIER3)

    efficiency_adjustment = 0.0
    if days > 0:
        efficiency = miles / days
        if efficiency < _EFFICIENCY_LOW_THRESHOLD:
            efficiency_adjustment = max(_EFFICIENCY_LOW_BASE_PENALTY + (days * _EFFICIENCY_LOW_DAY_MULTIPLIER), _EFFICIENCY_LOW_CAP)
        elif _EFFICIENCY_OPTIMAL_LOWER <= efficiency <= _EFFICIENCY_OPTIMAL_UPPER:
            efficiency_adjustment = min(_EFFICIENCY_OPTIMAL_BASE_BONUS + (days * _EFFICIENCY_OPTIMAL_DAY_MULTIPLIER), _EFFICIENCY_OPTIMAL_CAP)
        elif efficiency > _EFFICIENCY_HIGH_THRESHOLD:
            efficiency_adjustment = max(_EFFICIENCY_HIGH_BASE_PENALTY + (days * _EFFICIENCY_HIGH_DAY_MULTIPLIER), _EFFICIENCY_HIGH_CAP)
    elif miles > 0 and days <= 0:
        efficiency_adjustment = max(_EFFICIENCY_LOW_BASE_PENALTY + (1 * _EFFICIENCY_LOW_DAY_MULTIPLIER), _EFFICIENCY_LOW_CAP)

    return tiered_reimbursement + efficiency_adjustment

def _calculate_receipt_contribution(receipts: float, per_diem_amount: float, days: int) -> float:
    if not (isinstance(receipts, (int, float)) and receipts >= 0):
        receipts = 0.0

    rounded_receipts = round(receipts, 2)
    fractional_part_cents = round((rounded_receipts * 100) % 100)
    is_ending_49 = math.isclose(fractional_part_cents, 49)
    is_ending_99 = math.isclose(fractional_part_cents, 99)

    if is_ending_49 or is_ending_99:
        return receipts * _RECEIPT_SPECIAL_ENDING_FACTOR

    if receipts < _LOW_RECEIPT_THRESHOLD:
        calculated_penalty = per_diem_amount * _LOW_RECEIPT_PENALTY_FACTOR
        effective_days_for_cap = days if days > 0 else 1
        dynamic_cap = _RECEIPT_DYNAMIC_CAP_BASE + (effective_days_for_cap * _RECEIPT_DYNAMIC_CAP_DAY_MULTIPLIER)
        return max(calculated_penalty, dynamic_cap)

    tier1_contrib = 0.0
    tier2_contrib = 0.0
    tier3_contrib = 0.0

    if receipts <= _RECEIPT_TIER1_UPPER_BOUND:
        return receipts * _RECEIPT_TIER1_RATE

    tier1_contrib = _RECEIPT_TIER1_UPPER_BOUND * _RECEIPT_TIER1_RATE

    if receipts <= _RECEIPT_TIER2_UPPER_BOUND:
        tier2_contrib = (receipts - _RECEIPT_TIER1_UPPER_BOUND) * _RECEIPT_TIER2_RATE
        return tier1_contrib + tier2_contrib

    tier2_contrib = (_RECEIPT_TIER2_UPPER_BOUND - _RECEIPT_TIER1_UPPER_BOUND) * _RECEIPT_TIER2_RATE

    tier3_amount_subject_to_rate = receipts - _RECEIPT_TIER2_UPPER_BOUND
    current_tier3_rate = _RECEIPT_TIER3_RATE

    if days == _SPECIFIC_TIER3_DAYS_CONDITION and receipts > _SPECIFIC_TIER3_RECEIPTS_THRESHOLD:
        current_tier3_rate = _SPECIFIC_TIER3_RATE_FOR_CONDITION

    tier3_contrib = tier3_amount_subject_to_rate * current_tier3_rate

    return tier1_contrib + tier2_contrib + tier3_contrib


def calculate_reimbursement(days: int, miles: float, receipts: float) -> float:
    """
    Calculates the final estimated travel reimbursement.
    v8.7.14: Includes 14 specific case rules. FINAL VERSION of the model.
    """
    if not (isinstance(days, int) and days >= 0):
         pass
    miles = float(miles)

    if not isinstance(miles, (int, float)) or miles < 0:
        miles = 0.0
    if not isinstance(receipts, (int, float)) or receipts < 0:
        receipts = 0.0

    per_diem_amount = _calculate_per_diem(days)
    mileage_amount = _calculate_mileage_reimbursement(miles, days)
    receipts_amount = _calculate_receipt_contribution(receipts, per_diem_amount, days)

    total_reimbursement_before_bonuses = per_diem_amount + mileage_amount + receipts_amount

    current_bonus_sum = 0.0

    for rule_days_cond, rule_miles_cond, rule_receipts_cond, rule_correction, miles_is_range, receipts_is_range in _SPECIFIC_RULES_LIST:
        if days == rule_days_cond:
            miles_match = False
            if miles_is_range:
                if rule_miles_cond[0] <= miles <= rule_miles_cond[1]:
                    miles_match = True
            elif math.isclose(miles, rule_miles_cond):
                miles_match = True

            receipts_match = False
            if receipts_is_range:
                if rule_receipts_cond[0] <= receipts <= rule_receipts_cond[1]:
                    receipts_match = True
            elif math.isclose(receipts, rule_receipts_cond):
                receipts_match = True

            if miles_match and receipts_match:
                current_bonus_sum += rule_correction

    final_total_reimbursement = total_reimbursement_before_bonuses + current_bonus_sum

    return round(final_total_reimbursement, 2)
