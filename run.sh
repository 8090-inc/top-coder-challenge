#!/usr/bin/env bash
# ACME Reimbursement Engine - Reverse-engineered implementation v4 (Data-Driven)
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
# Requires: bash 4+, bc
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <days> <miles> <receipts>" >&2
  exit 1
fi

D="$1"        # trip_duration_days
M="$2"        # miles_traveled  
R="$3"        # total_receipts_amount

#—— BASE COMPONENT: Simple per-day declining rate ————————————————
# Analysis shows per-day rates decline with trip length
if (( $(echo "$D == 1" | bc -l) )); then
  base=$(echo "scale=4; 100" | bc)  # 1-day trips have different logic
elif (( $(echo "$D <= 3" | bc -l) )); then
  base=$(echo "scale=4; $D * 100" | bc)
elif (( $(echo "$D <= 7" | bc -l) )); then
  base=$(echo "scale=4; $D * 95" | bc)
elif (( $(echo "$D <= 12" | bc -l) )); then
  base=$(echo "scale=4; $D * 85" | bc)
else
  base=$(echo "scale=4; $D * 75" | bc)
fi

#—— MILEAGE COMPONENT: Higher base rate with tiering ——————————————
# Analysis found ~$0.76/mile base rate from low-receipt cases
if (( $(echo "$M <= 200" | bc -l) )); then
  mileage=$(echo "scale=4; $M * 0.75" | bc)
elif (( $(echo "$M <= 500" | bc -l) )); then
  base_200=$(echo "scale=4; 200 * 0.75" | bc)
  excess=$(echo "scale=4; $M - 200" | bc)
  mileage=$(echo "scale=4; $base_200 + ($excess * 0.60)" | bc)
elif (( $(echo "$M <= 1000" | bc -l) )); then
  base_200=$(echo "scale=4; 200 * 0.75" | bc)
  mid_300=$(echo "scale=4; 300 * 0.60" | bc)
  excess=$(echo "scale=4; $M - 500" | bc)
  mileage=$(echo "scale=4; $base_200 + $mid_300 + ($excess * 0.45)" | bc)
else
  base_200=$(echo "scale=4; 200 * 0.75" | bc)
  mid_300=$(echo "scale=4; 300 * 0.60" | bc)
  high_500=$(echo "scale=4; 500 * 0.45" | bc)
  excess=$(echo "scale=4; $M - 1000" | bc)
  mileage=$(echo "scale=4; $base_200 + $mid_300 + $high_500 + ($excess * 0.25)" | bc)
fi

#—— RECEIPT COMPONENT: Complex tiering based on analysis ——————————
# Analysis shows receipts are primary component with complex multipliers
if (( $(echo "$R < 10" | bc -l) )); then
  # Very low receipts get huge multipliers (5.56x average)
  net_receipts=$(echo "scale=4; $R * 5.0" | bc)
elif (( $(echo "$R < 50" | bc -l) )); then
  # Low receipts get good multipliers (1.58x average)
  net_receipts=$(echo "scale=4; $R * 1.8" | bc)
elif (( $(echo "$R < 100" | bc -l) )); then
  # Medium-low receipts get modest bonus
  net_receipts=$(echo "scale=4; $R * 1.2" | bc)
elif (( $(echo "$R < 200" | bc -l) )); then
  # Medium receipts get standard rate
  net_receipts=$(echo "scale=4; $R * 1.0" | bc)
elif (( $(echo "$R < 500" | bc -l) )); then
  # Medium-high receipts get slight bonus
  net_receipts=$(echo "scale=4; $R * 0.9" | bc)
elif (( $(echo "$R < 1000" | bc -l) )); then
  # High receipts get good treatment (0.42x analysis result)
  net_receipts=$(echo "scale=4; $R * 0.6" | bc)
elif (( $(echo "$R < 2000" | bc -l) )); then
  # Very high gets moderate treatment (0.50x analysis result)
  net_receipts=$(echo "scale=4; $R * 0.5" | bc)
else
  # Extreme receipts get conservative treatment (0.30x analysis result)
  base_2000=$(echo "scale=4; 2000 * 0.5" | bc)
  excess=$(echo "scale=4; $R - 2000" | bc)
  net_receipts=$(echo "scale=4; $base_2000 + ($excess * 0.3)" | bc)
fi

#—— SPECIAL LOGIC FOR 1-DAY TRIPS ——————————————————————————————
# 1-day trips with extreme mileage get capped
if (( $(echo "$D == 1" | bc -l) )); then
  mpd=$(echo "scale=4; $M" | bc)  # MPD = miles for 1-day trips
  
  # Apply mileage caps for extreme 1-day trips
  if (( $(echo "$mpd > 800" | bc -l) )); then
    # Extreme mileage gets heavy penalty
    mileage=$(echo "scale=4; $mileage * 0.6" | bc)
  elif (( $(echo "$mpd > 500" | bc -l) )); then
    # High mileage gets moderate penalty
    mileage=$(echo "scale=4; $mileage * 0.8" | bc)
  fi
  
  # High receipt 1-day trips get capped too
  if (( $(echo "$R > 1500 && $mpd > 500" | bc -l) )); then
    # Both high receipts AND high mileage = major cap
    raw_total=$(echo "scale=4; $base + $mileage + $net_receipts" | bc)
    if (( $(echo "$raw_total > 1400" | bc -l) )); then
      raw_total=$(echo "scale=4; 1400" | bc)
    fi
  fi
fi

#—— WINDFALL ROUNDING (.49/.99 cents) ——————————————————————————
cents=$(printf "%.2f" "$R" | awk -F'.' '{print $2}')
windfall=0
if [[ "$cents" == "49" || "$cents" == "99" ]]; then
  windfall=$(echo "scale=4; 3" | bc)
fi

#—— FINAL CALCULATION ————————————————————————————————————————
if [[ -z "${raw_total:-}" ]]; then
  raw_total=$(echo "scale=4; $base + $mileage + $net_receipts + $windfall" | bc)
else
  raw_total=$(echo "scale=4; $raw_total + $windfall" | bc)
fi

# Ensure non-negative result
if (( $(echo "$raw_total < 0" | bc -l) )); then
  raw_total=0
fi

# Round to 2 decimal places
reimbursement=$(printf "%.2f" "$raw_total")
echo "$reimbursement" 