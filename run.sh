#!/bin/bash
trip_duration_days=$1
miles_traveled=$2
total_receipts_amount=$3

# Ensure inputs are numeric for bc operations
trip_duration_days_val=$(echo "$trip_duration_days" | bc)
miles_traveled_val=$(echo "$miles_traveled" | bc)
receipts_val=$(echo "$total_receipts_amount" | bc)

# Per Diem
per_diem_reimbursement=$(echo "$trip_duration_days_val * 100" | bc)

# Mileage
mileage_reimbursement=0
if (( $(echo "$miles_traveled_val <= 100" | bc -l) )); then
  mileage_reimbursement=$(echo "$miles_traveled_val * 0.58" | bc)
else
  over_100_miles=$(echo "$miles_traveled_val - 100" | bc)
  mileage_reimbursement=$(echo "(100 * 0.58) + ($over_100_miles * 0.30)" | bc)
fi

# Receipts
receipt_component=0
low_receipt_threshold=20
is_low_receipt=0
if (( $(echo "$trip_duration_days_val > 1" | bc -l) )) && (( $(echo "$receipts_val < $low_receipt_threshold" | bc -l) )); then
  is_low_receipt=1
fi

if [ "$is_low_receipt" -eq 1 ]; then
  receipt_component=0
else
  receipt_percentage=0.70
  current_max_receipt_reimbursement=500 # Default
  if (( $(echo "$trip_duration_days_val > 7" | bc -l) )); then
    current_max_receipt_reimbursement=250 # Lower cap for trips longer than 7 days
  fi

  potential_receipt_reimbursement=$(echo "$receipts_val * $receipt_percentage" | bc)

  if (( $(echo "$potential_receipt_reimbursement < $current_max_receipt_reimbursement" | bc -l) )); then
    receipt_component=$potential_receipt_reimbursement
  else
    receipt_component=$current_max_receipt_reimbursement
  fi
fi

# Trip Duration Bonus (using integer comparison for trip_duration_days)
trip_duration_days_int=$(printf "%.0f" "$trip_duration_days_val") # Convert to int for bash comparison
duration_bonus=0
if [ "$trip_duration_days_int" -eq 5 ]; then
  duration_bonus=50
elif [ "$trip_duration_days_int" -eq 4 ] || [ "$trip_duration_days_int" -eq 6 ]; then
  duration_bonus=25
fi

# MPD Efficiency Bonus/Penalty
mpd_bonus=0
if (( $(echo "$trip_duration_days_val > 0" | bc -l) )); then
  # scale=2 for division precision
  mpd=$(echo "scale=2; $miles_traveled_val / $trip_duration_days_val" | bc)
  if (( $(echo "$mpd >= 180 && $mpd <= 220" | bc -l) )); then
    mpd_bonus=75
  elif (( $(echo "$mpd < 50" | bc -l) )); then
    mpd_bonus=-50
  elif (( $(echo "$mpd > 350" | bc -l) )); then
    mpd_bonus=-25
  fi
fi

total_reimbursement=$(echo "$per_diem_reimbursement + $mileage_reimbursement + $receipt_component + $duration_bonus + $mpd_bonus" | bc)
printf "%.2f\n" $total_reimbursement