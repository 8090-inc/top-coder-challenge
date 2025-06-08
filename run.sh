#!/bin/bash

#!/bin/bash

# Read the input parameters
days="$1"
miles="$2"
receipts_val="$3"

# --- Default mileage calculation (M2 rate is 0.45) ---
if (( $(echo "$miles <= 100" | bc -l) )); then
  miles_calc=$(echo "scale=2; $miles * 0.58" | bc)
else
  miles_calc=$(echo "scale=2; (100 * 0.58) + (($miles - 100) * 0.45)" | bc)
fi

# Initialize variables
per_diem_calc="0.00"
calculated_rcpt_component="0.00"
bonus_5_day="0.00"

# --- Core Logic (from score-19635 model) ---
if (( $(echo "$receipts_val < 15" | bc -l) )); then
  # --- Path A: Very Low Receipts ---
  per_diem_calc=$(echo "scale=2; ($days * 108.20) - 14.00" | bc)
  # calculated_rcpt_component remains 0.00
  # bonus_5_day remains 0.00
else
  # --- Path B: receipts_val >= 15 ---

  # Duration-dependent Per Diem for Path B
  if (( $(echo "$days <= 3" | bc -l) )); then
    per_diem_calc=$(echo "scale=2; $days * 100.00" | bc)
  elif (( $(echo "$days >= 4 && $days <= 6" | bc -l) )); then
    per_diem_calc=$(echo "scale=2; $days * 90.00" | bc)
  else # days >= 7
    per_diem_calc=$(echo "scale=2; $days * 85.00" | bc)
  fi

  # Tiered Receipt Reimbursement for Path B
  r_val=$(echo "scale=2; $receipts_val / 1.00" | bc) # Ensure r_val is scale=2
  # calculated_rcpt_component is already 0.00

  receipt_tier1_max="200.00"; rate1="0.60"
  tier1_contrib_receipts=$(echo "scale=2; if (${r_val} < ${receipt_tier1_max}) ${r_val} else ${receipt_tier1_max}" | bc)
  calculated_rcpt_component=$(echo "scale=2; ${calculated_rcpt_component} + ${tier1_contrib_receipts} * ${rate1}" | bc)
  r_val=$(echo "scale=2; ${r_val} - ${tier1_contrib_receipts}" | bc)

  receipt_tier2_upto="1000.00"; rate2="0.40"
  tier2_reimbursement_limit=$(echo "scale=2; ${receipt_tier2_upto} - ${receipt_tier1_max}" | bc)

  is_r_positive_tier2=$(echo "${r_val} > 0" | bc -l)
  if [ "$is_r_positive_tier2" -eq 1 ]; then
    tier2_contrib_receipts=$(echo "scale=2; if (${r_val} < ${tier2_reimbursement_limit}) ${r_val} else ${tier2_reimbursement_limit}" | bc)
    calculated_rcpt_component=$(echo "scale=2; ${calculated_rcpt_component} + ${tier2_contrib_receipts} * ${rate2}" | bc)
    r_val=$(echo "scale=2; ${r_val} - ${tier2_contrib_receipts}" | bc)
  fi

  rate3="0.20"
  is_r_positive_tier3=$(echo "${r_val} > 0" | bc -l)
  if [ "$is_r_positive_tier3" -eq 1 ]; then
    calculated_rcpt_component=$(echo "scale=2; ${calculated_rcpt_component} + ${r_val} * ${rate3}" | bc)
  fi

  # 5-Day Bonus for Path B
  if (( $(echo "$days == 5" | bc -l) )); then
    bonus_5_day=$(echo "scale=2; 75.00" | bc)
  fi
fi

# --- Initial Total Reimbursement ---
initial_total_reimbursement=$(echo "scale=2; $per_diem_calc + $miles_calc + $calculated_rcpt_component + $bonus_5_day" | bc)

# --- Extreme Over-reimbursement Adjustment ---
final_total_reimbursement="$initial_total_reimbursement"

is_long_trip=$(echo "${days} >= 7" | bc -l) # Use -l for bc comparison
is_high_receipts=$(echo "${receipts_val} > 1000" | bc -l) # Use -l for bc comparison

if [ "$is_long_trip" -eq 1 ] && [ "$is_high_receipts" -eq 1 ]; then
  max_rcpt_component_for_extreme_case=$(echo "scale=2; $days * 50.00" | bc)
  is_rcpt_over_limit=$(echo "${calculated_rcpt_component} > ${max_rcpt_component_for_extreme_case}" | bc -l) # Use -l

  if [ "$is_rcpt_over_limit" -eq 1 ]; then
    reduction_amount=$(echo "scale=2; ${calculated_rcpt_component} - ${max_rcpt_component_for_extreme_case}" | bc)
    final_total_reimbursement=$(echo "scale=2; ${initial_total_reimbursement} - ${reduction_amount}" | bc)
  fi
fi

# Output the final result
printf "%.2f\n" "$final_total_reimbursement"