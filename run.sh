#!/bin/bash

# Read the input parameters
days="$1"
miles="$2"
receipts_val="$3"

# --- Standard Mileage Calculation (used across paths) ---
if (( $(echo "$miles <= 100" | bc -l) )); then
  miles_calc_std=$(echo "scale=2; $miles * 0.58" | bc)
else
  miles_calc_std=$(echo "scale=2; (100 * 0.58) + (($miles - 100) * 0.45)" | bc)
fi

# Initialize variables that will be set in different paths
per_diem_calc="0.00"
rcpt_calc="0.00"
miles_calc="$miles_calc_std" # Default to standard, can be overridden by Path X
bonus_5_day="0.00"
total_reimbursement="0.00"

# --- Path Logic ---
if (( $(echo "$receipts_val < 15" | bc -l) )); then
  # --- Path A: Very Low Receipts ---
  per_diem_calc=$(echo "scale=2; ($days * 108.20) - 14.00" | bc)
  # rcpt_calc is already 0.00
  # bonus_5_day is already 0.00
  # miles_calc is already miles_calc_std
  total_reimbursement=$(echo "scale=2; $per_diem_calc + $miles_calc + $rcpt_calc + $bonus_5_day" | bc)
else
  # --- Not Path A (receipts_val >= 15), Check for "Extreme Outlier" or Path B ---
  avg_daily_receipts=$(echo "scale=2; if($days > 0) { $receipts_val / $days } else { 0 }" | bc -l)
  miles_per_day=$(echo "scale=2; if($days > 0) { $miles / $days } else { 0 }" | bc -l)

  # Conditions for "Case 684-like" Extreme Outlier:
  cond_days=$(echo "${days} >= 7" | bc -l)
  cond_receipts=$(echo "${avg_daily_receipts} > 200.00" | bc -l)
  cond_miles_day=$(echo "${miles_per_day} < 100.00" | bc -l)

  if [ "$cond_days" -eq 1 ] && [ "$cond_receipts" -eq 1 ] && [ "$cond_miles_day" -eq 1 ]; then
    # --- Extreme Outlier Path ---
    per_diem_calc=$(echo "scale=2; $days * 55.00" | bc)

    miles_raw="$miles_calc_std" # Start with standard mileage
    miles_cap="100.00"
    miles_calc=$(echo "scale=2; if (${miles_raw} < ${miles_cap}) ${miles_raw} else ${miles_cap}" | bc)

    rcpt_raw=$(echo "scale=2; $receipts_val * 0.10" | bc)
    rcpt_cap="100.00"
    rcpt_calc=$(echo "scale=2; if (${rcpt_raw} < ${rcpt_cap}) ${rcpt_raw} else ${rcpt_cap}" | bc)

    bonus_5_day="0.00" # No 5-day bonus for these extreme trips
    final_total_reimbursement=$(echo "scale=2; $per_diem_calc + $miles_calc + $rcpt_calc + $bonus_5_day" | bc)
    # Path taken, assign to total_reimbursement for final print
    total_reimbursement="$final_total_reimbursement"
  else
    # --- Path B: Standard logic for receipts_val >= 15 (and not Extreme Outlier) ---
    # miles_calc is already miles_calc_std (set by default)

    # Duration-dependent Per Diem for Path B
    if (( $(echo "$days <= 3" | bc -l) )); then
      per_diem_calc=$(echo "scale=2; $days * 100.00" | bc)
    elif (( $(echo "$days >= 4 && $days <= 6" | bc -l) )); then
      per_diem_calc=$(echo "scale=2; $days * 90.00" | bc)
    else # days >= 7 (and not Extreme Outlier)
      per_diem_calc=$(echo "scale=2; $days * 85.00" | bc)
    fi

    # Tiered Receipt Reimbursement for Path B
    r_val_b=$(echo "scale=2; $receipts_val / 1.00" | bc) # Use _b suffix for Path B specific temp vars
    rcpt_calc_b="0.00"

    r1_max_b="200.00"; rt1_b="0.60"
    t1_contrib_b=$(echo "scale=2; if (${r_val_b} < ${r1_max_b}) ${r_val_b} else ${r1_max_b}" | bc)
    rcpt_calc_b=$(echo "scale=2; ${rcpt_calc_b} + ${t1_contrib_b} * ${rt1_b}" | bc)
    r_val_b=$(echo "scale=2; ${r_val_b} - ${t1_contrib_b}" | bc)

    r2_upto_b="1000.00"; rt2_b="0.40"
    r2_limit_b=$(echo "scale=2; ${r2_upto_b} - ${r1_max_b}" | bc)

    is_r_pos1_b=$(echo "${r_val_b} > 0" | bc -l)
    if [ "$is_r_pos1_b" -eq 1 ]; then
      t2_contrib_b=$(echo "scale=2; if (${r_val_b} < ${r2_limit_b}) ${r_val_b} else ${r2_limit_b}" | bc)
      rcpt_calc_b=$(echo "scale=2; ${rcpt_calc_b} + ${t2_contrib_b} * ${rt2_b}" | bc)
      r_val_b=$(echo "scale=2; ${r_val_b} - ${t2_contrib_b}" | bc)
    fi

    rt3_b="0.20"
    is_r_pos2_b=$(echo "${r_val_b} > 0" | bc -l)
    if [ "$is_r_pos2_b" -eq 1 ]; then
      rcpt_calc_b=$(echo "scale=2; ${rcpt_calc_b} + ${r_val_b} * ${rt3_b}" | bc)
    fi
    rcpt_calc="$rcpt_calc_b" # Assign to the common variable used in final sum

    # 5-Day Bonus for Path B
    bonus_5_day="0.00" # Reset for Path B
    if (( $(echo "$days == 5" | bc -l) )); then
      bonus_5_day=$(echo "scale=2; 75.00" | bc)
    fi
    final_total_reimbursement=$(echo "scale=2; $per_diem_calc + $miles_calc + $rcpt_calc + $bonus_5_day" | bc)
    # Path taken, assign to total_reimbursement for final print
    total_reimbursement="$final_total_reimbursement"
  fi
fi

# Output the final result
printf "%.2f\n" "$total_reimbursement"