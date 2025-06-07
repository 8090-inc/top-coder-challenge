#!/usr/bin/env bash
# ACME Reimbursement Engine - Reverse-engineered implementation v3
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

#—— 1-DAY TRIPS: Special Logic (Heavily Capped) ———————————————————
if (( $(echo "$D == 1" | bc -l) )); then
  # Base for 1-day trips
  base=$(echo "scale=4; 120" | bc)
  
  # Mileage with heavy cap (observed: never more than ~$150 for mileage)
  mileage=$(echo "scale=4; $M * 0.25" | bc)
  if (( $(echo "$mileage > 150" | bc -l) )); then
    mileage=$(echo "scale=4; 150" | bc)
  fi
  
  # Receipt processing (very complex for 1-day trips)
  if (( $(echo "$R < 20" | bc -l) )); then
    # Very low receipts get penalties
    net_receipts=$(echo "scale=4; $R * 0.3" | bc)
  elif (( $(echo "$R < 100" | bc -l) )); then
    # Low receipts get modest boost
    net_receipts=$(echo "scale=4; $R * 1.2" | bc)
  elif (( $(echo "$R < 500" | bc -l) )); then
    # Medium receipts get decent treatment
    net_receipts=$(echo "scale=4; $R * 0.7" | bc)
  elif (( $(echo "$R < 1500" | bc -l) )); then
    # High receipts start getting heavily penalized
    base_500=$(echo "scale=4; 500 * 0.7" | bc)
    excess=$(echo "scale=4; $R - 500" | bc)
    net_receipts=$(echo "scale=4; $base_500 + ($excess * 0.4)" | bc)
  else
    # Very high receipts get massive penalties
    base_500=$(echo "scale=4; 500 * 0.7" | bc)
    mid_1000=$(echo "scale=4; 1000 * 0.4" | bc)
    excess=$(echo "scale=4; $R - 1500" | bc)
    net_receipts=$(echo "scale=4; $base_500 + $mid_1000 + ($excess * 0.1)" | bc)
  fi

#—— MULTI-DAY TRIPS: Different Logic ——————————————————————————————
else
  # Base scales with trip length but with diminishing returns
  if (( $(echo "$D <= 3" | bc -l) )); then
    base=$(echo "scale=4; $D * 70" | bc)
  elif (( $(echo "$D <= 7" | bc -l) )); then
    base=$(echo "scale=4; $D * 55" | bc)
  elif (( $(echo "$D <= 14" | bc -l) )); then
    base=$(echo "scale=4; $D * 45" | bc)
  else
    base=$(echo "scale=4; $D * 40" | bc)
  fi
  
  # Mileage component with MODEST efficiency bonuses for high MPD
  mpd=$(echo "scale=4; $M / $D" | bc)
  
  if (( $(echo "$mpd >= 120" | bc -l) )); then
    # Very high efficiency gets modest bonus
    mileage=$(echo "scale=4; $M * 0.5" | bc)
  elif (( $(echo "$mpd >= 80" | bc -l) )); then
    # High efficiency gets small bonus
    mileage=$(echo "scale=4; $M * 0.4" | bc)
  else
    # Standard rate for most cases
    mileage=$(echo "scale=4; $M * 0.3" | bc)
  fi
  
  # Receipt processing with EXTREME penalties for high amounts
  if (( $(echo "$R < 100" | bc -l) )); then
    net_receipts=$(echo "scale=4; $R * 1.0" | bc)
  elif (( $(echo "$R < 600" | bc -l) )); then
    net_receipts=$(echo "scale=4; $R * 0.6" | bc)
  elif (( $(echo "$R < 1500" | bc -l) )); then
    # Heavy penalties for high receipts
    base_600=$(echo "scale=4; 600 * 0.6" | bc)
    excess=$(echo "scale=4; $R - 600" | bc)
    net_receipts=$(echo "scale=4; $base_600 + ($excess * 0.1)" | bc)
  else
    # Massive penalties for very high receipts
    base_600=$(echo "scale=4; 600 * 0.6" | bc)
    mid_900=$(echo "scale=4; 900 * 0.1" | bc)
    excess=$(echo "scale=4; $R - 1500" | bc)
    net_receipts=$(echo "scale=4; $base_600 + $mid_900 + ($excess * 0.02)" | bc)
  fi
fi

#—— WINDFALL ROUNDING (.49/.99 cents) ——————————————————————————
cents=$(printf "%.2f" "$R" | awk -F'.' '{print $2}')
windfall=0
if [[ "$cents" == "49" || "$cents" == "99" ]]; then
  windfall=$(echo "scale=4; 3" | bc)
fi

#—— FINAL CALCULATION ————————————————————————————————————————
raw_total=$(echo "scale=4; $base + $mileage + $net_receipts + $windfall" | bc)

# Ensure non-negative result
if (( $(echo "$raw_total < 0" | bc -l) )); then
  raw_total=0
fi

# Round to 2 decimal places
reimbursement=$(printf "%.2f" "$raw_total")
echo "$reimbursement" 