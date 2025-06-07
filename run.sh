#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <days> <miles> <receipts>" >&2
  exit 1
fi

DAYS="$1"
MILES="$2"
RECEIPTS="$3"

# Constants from Python pseudocode
FEE="80.00"
PER_DIEM_STANDARD="100.00"
PER_DIEM_LONG_TRIP="90.00"
DAY_5_BONUS_RATE="25.00"
DAY_8_BONUS="120.00"
CENTS_BUG_BONUS="5.00"
EFFICIENCY_BONUS_RATE="0.05"
EFFICIENCY_THRESHOLD="150"

# RECEIPT_RULES (amount_threshold, rate)
# (100, 1.00), (500, 0.41), (1000, 0.74), (inf, 0.46)
RECEIPT_RULE_1_THRESH="100"
RECEIPT_RULE_1_RATE="1.00"
RECEIPT_RULE_2_THRESH="500" # up to 500 (i.e., next 400)
RECEIPT_RULE_2_RATE="0.41"
RECEIPT_RULE_3_THRESH="1000" # up to 1000 (i.e., next 500)
RECEIPT_RULE_3_RATE="0.74"
RECEIPT_RULE_LAST_RATE="0.46" # for amounts > 1000

# MILEAGE_RATES (miles_threshold, rate)
# (100, 0.58), (500, 0.45), (inf, 0.35)
MILEAGE_RATE_1_THRESH="100"
MILEAGE_RATE_1_RATE="0.58"
MILEAGE_RATE_2_THRESH="500" # up to 500 (i.e., next 400)
MILEAGE_RATE_2_RATE="0.45"
MILEAGE_RATE_LAST_RATE="0.35" # for amounts > 500

awk -v days="$DAYS" \
    -v miles="$MILES" \
    -v receipts="$RECEIPTS" \
    -v fee="$FEE" \
    -v per_diem_standard="$PER_DIEM_STANDARD" \
    -v per_diem_long_trip="$PER_DIEM_LONG_TRIP" \
    -v day_5_bonus_rate="$DAY_5_BONUS_RATE" \
    -v day_8_bonus="$DAY_8_BONUS" \
    -v cents_bug_bonus="$CENTS_BUG_BONUS" \
    -v efficiency_bonus_rate="$EFFICIENCY_BONUS_RATE" \
    -v efficiency_threshold="$EFFICIENCY_THRESHOLD" \
    -v receipt_rule_1_thresh="$RECEIPT_RULE_1_THRESH" \
    -v receipt_rule_1_rate="$RECEIPT_RULE_1_RATE" \
    -v receipt_rule_2_thresh="$RECEIPT_RULE_2_THRESH" \
    -v receipt_rule_2_rate="$RECEIPT_RULE_2_RATE" \
    -v receipt_rule_3_thresh="$RECEIPT_RULE_3_THRESH" \
    -v receipt_rule_3_rate="$RECEIPT_RULE_3_RATE" \
    -v receipt_rule_last_rate="$RECEIPT_RULE_LAST_RATE" \
    -v mileage_rate_1_thresh="$MILEAGE_RATE_1_THRESH" \
    -v mileage_rate_1_rate="$MILEAGE_RATE_1_RATE" \
    -v mileage_rate_2_thresh="$MILEAGE_RATE_2_THRESH" \
    -v mileage_rate_2_rate="$MILEAGE_RATE_2_RATE" \
    -v mileage_rate_last_rate="$MILEAGE_RATE_LAST_RATE" \
'BEGIN {
    # 1. Initialize reimbursement with FEE
    reimbursement = fee;

    # 2. Per Diem calculation
    per_diem_total = 0;
    if (days >= 8) {
        per_diem_total = days * per_diem_long_trip;
    } else {
        per_diem_total = days * per_diem_standard;
    }
    if (days == 5) {
        per_diem_total += days * day_5_bonus_rate;
    }
    if (days == 8) { # Note: Python pseudocode has this as days == 8, not days >= 8
        per_diem_total += day_8_bonus;
    }
    reimbursement += per_diem_total;

    # 3. Mileage calculation
    mileage_reimbursement = 0;
    miles_remaining = miles;

    # Rate 1: up to mileage_rate_1_thresh
    if (miles_remaining > 0) {
        chargeable_miles = (miles_remaining < mileage_rate_1_thresh) ? miles_remaining : mileage_rate_1_thresh;
        mileage_reimbursement += chargeable_miles * mileage_rate_1_rate;
        miles_remaining -= chargeable_miles;
    }
    # Rate 2: up to mileage_rate_2_thresh (excess over rate 1)
    if (miles_remaining > 0) {
        limit_for_rate_2 = mileage_rate_2_thresh - mileage_rate_1_thresh;
        chargeable_miles = (miles_remaining < limit_for_rate_2) ? miles_remaining : limit_for_rate_2;
        mileage_reimbursement += chargeable_miles * mileage_rate_2_rate;
        miles_remaining -= chargeable_miles;
    }
    # Rate 3 (last rate): for remaining miles
    if (miles_remaining > 0) {
        mileage_reimbursement += miles_remaining * mileage_rate_last_rate;
    }

    # New Efficiency Bonus Logic
    if (days > 0) {
        miles_per_day = miles / days;
        # Order of conditions matters here
        if (miles_per_day >= 180 && miles_per_day <= 220) {
            mileage_reimbursement *= 1.15; # Max bonus for sweet spot (15% bonus)
        } else if (miles_per_day > 220 && miles_per_day < 300) { # Tapering bonus
            mileage_reimbursement *= 1.05; # (5% bonus)
        } else if (miles_per_day < 100 && miles_per_day > 0) { # miles_per_day > 0 to avoid issues with 0 miles
            mileage_reimbursement *= 0.80; # Penalty for low efficiency (20% penalty)
        } else if (miles_per_day >= 300) {
            mileage_reimbursement *= 0.90; # Penalty for very high efficiency (10% penalty)
        }
        # Trips with 100 <= miles_per_day < 180 will have no specific adjustment from this block.
        # The variables efficiency_threshold and efficiency_bonus_rate are no longer used by this new logic.
    }
    reimbursement += mileage_reimbursement;

    # 4. Receipts calculation
    receipt_reimbursement = 0;
    receipts_remaining = receipts;

    # Rule 1: up to receipt_rule_1_thresh
    if (receipts_remaining > 0) {
        chargeable_receipts = (receipts_remaining < receipt_rule_1_thresh) ? receipts_remaining : receipt_rule_1_thresh;
        receipt_reimbursement += chargeable_receipts * receipt_rule_1_rate;
        receipts_remaining -= chargeable_receipts;
    }
    # Rule 2: up to receipt_rule_2_thresh (excess over rule 1)
    if (receipts_remaining > 0) {
        limit_for_rule_2 = receipt_rule_2_thresh - receipt_rule_1_thresh; # Next 400
        chargeable_receipts = (receipts_remaining < limit_for_rule_2) ? receipts_remaining : limit_for_rule_2;
        receipt_reimbursement += chargeable_receipts * receipt_rule_2_rate;
        receipts_remaining -= chargeable_receipts;
    }
    # Rule 3: up to receipt_rule_3_thresh (excess over rule 2)
    if (receipts_remaining > 0) {
        limit_for_rule_3 = receipt_rule_3_thresh - receipt_rule_2_thresh; # Next 500
        chargeable_receipts = (receipts_remaining < limit_for_rule_3) ? receipts_remaining : limit_for_rule_3;
        receipt_reimbursement += chargeable_receipts * receipt_rule_3_rate;
        receipts_remaining -= chargeable_receipts;
    }
    # Rule 4 (last rate): for remaining receipts
    if (receipts_remaining > 0) {
        receipt_reimbursement += receipts_remaining * receipt_rule_last_rate;
    }

    # Small receipts penalty
    if (receipts > 0 && receipts < 25) {
        receipt_reimbursement *= 0.5;
    }
    reimbursement += receipt_reimbursement;

    # 5. Cents Bug Bonus
    # Awk rounds, so 10.49 * 100 = 1049. 10.99 * 100 = 1099.
    # We need to be careful with floating point inaccuracies for modulus.
    # A common way is to convert to integer cents by rounding.
    receipt_cents_exact = receipts * 100;
    # Adding 0.00001 to handle potential floating point representation issues before int()
    # For example, 10.49 could be 10.489999999999999...
    # int(10.49 * 100) might be 1048. int(10.49 * 100 + 0.5) for rounding is better.
    receipt_cents_int = int(receipt_cents_exact + 0.00001);
    # actual_cents = receipt_cents_int % 100; # This line was commented out in draft

    # A more robust way for cents, given printf is available in awk:
    # Use sprintf to format receipts to two decimal places, then extract cents.
    formatted_receipts = sprintf("%.2f", receipts);
    split_receipts_count = split(formatted_receipts, parts, ".");
    if (split_receipts_count == 2) {
        actual_cents_str = parts[2];
        # Convert string cents to number for comparison
        actual_cents_val = actual_cents_str + 0;
        if (actual_cents_val == 49 || actual_cents_val == 99) {
            reimbursement += cents_bug_bonus;
        }
    } else if (receipts > 0 && (int(receipts) == receipts)) {
        # Handle whole numbers, cents are 00
        # No bonus for .00
    }


    # 6. Final rounding
    # Using printf for rounding to 2 decimal places
    printf "%.2f\n", reimbursement;
}'