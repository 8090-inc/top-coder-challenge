#!/usr/bin/env bash
# ACME Reimbursement Engine - Reverse-engineered implementation
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

#—— 1. Base per-diem calculation ————————————————————
# Base rate appears to be around $100-130/day depending on trip characteristics
base=$(echo "scale=4; 110 * $D" | bc)

# Efficiency bonus calculation (miles per day)
mpd=$(echo "scale=4; $M / $D" | bc)
efficiency_bonus=0

# Efficiency sweet spot bonus (150-220 MPD gets significant bonus)
if (( $(echo "$mpd >= 150 && $mpd <= 220" | bc -l) )); then
  efficiency_bonus=$(echo "scale=4; $base * 0.15" | bc)
fi

#—— 2. Mileage component (tiered rates) ——————————————————
if (( $(echo "$M <= 100" | bc -l) )); then
  rate=1.24
elif (( $(echo "$M <= 500" | bc -l) )); then
  # Split calculation: first 100 at high rate, remainder at mid rate
  first_100=$(echo "scale=4; 100 * 1.24" | bc)
  remainder=$(echo "scale=4; ($M - 100) * 0.37" | bc)
  m_component=$(echo "scale=4; $first_100 + $remainder" | bc)
else
  # Split calculation: first 100 + next 400 + remainder
  first_100=$(echo "scale=4; 100 * 1.24" | bc)  
  next_400=$(echo "scale=4; 400 * 0.37" | bc)
  remainder=$(echo "scale=4; ($M - 500) * 0.29" | bc)
  m_component=$(echo "scale=4; $first_100 + $next_400 + $remainder" | bc)
fi

# Simple rate for <=100 case
if (( $(echo "$M <= 100" | bc -l) )); then
  m_component=$(echo "scale=4; $rate * $M" | bc)
fi

#—— 3. Receipt processing (diminishing returns) ————————————————
# Small receipt penalty
receipt_penalty=0
per_day_receipts=$(echo "scale=4; $R / $D" | bc)
if (( $(echo "$per_day_receipts < 25" | bc -l) )); then
  receipt_penalty=$(echo "scale=4; 30" | bc)
fi

# Receipt calculation with diminishing returns
if (( $(echo "$R <= 500" | bc -l) )); then
  net_receipts=$(echo "scale=4; 0.85 * $R" | bc)
else
  above=$(echo "scale=4; $R - 500" | bc)
  net_receipts=$(echo "scale=4; (0.85 * 500) + (0.25 * $above)" | bc)
fi

# Rounding windfall (+$3 for receipts ending in .49 or .99)
cents=$(printf "%.2f" "$R" | awk -F'.' '{print $2}')
windfall=0
if [[ "$cents" == "49" || "$cents" == "99" ]]; then
  windfall=3
fi

#—— 4. Special adjustments ——————————————————————————————
# High mileage penalty adjustment
high_mileage_penalty=0
if (( $(echo "$M > 400 && $R > 1000" | bc -l) )); then
  high_mileage_penalty=$(echo "scale=4; 50" | bc)
fi

#—— 5. Final calculation ————————————————————————————————
raw_total=$(echo "scale=4; $base + $efficiency_bonus + $m_component + $net_receipts - $receipt_penalty + $windfall - $high_mileage_penalty" | bc)

# Ensure non-negative result
if (( $(echo "$raw_total < 0" | bc -l) )); then
  raw_total=0
fi

# Round to 2 decimal places
reimbursement=$(printf "%.2f" "$raw_total")
echo "$reimbursement" 