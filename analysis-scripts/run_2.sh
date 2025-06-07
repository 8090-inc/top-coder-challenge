#!/usr/bin/env bash
# ACME Reimbursement Engine - parity implementation (v0.9)
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
# Example: ./run.sh 5 730 1248.99
# Requires: bash 4+, bc
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <days> <miles> <receipts>" >&2
  exit 1
fi

D="$1"        # trip_duration_days (integer)
M="$2"        # miles_traveled (integer)
R="$3"        # total_receipts_amount (float, e.g. 1248.99)

#—— 1. Base per-diem ————————————————————————————
base=$(echo "scale=4; 100 * $D" | bc)
if [[ "$D" -eq 5 ]]; then
  base=$(echo "scale=4; $base + 50" | bc)  # 5-day bump
fi

#—— 2. Mileage component —————————————————————————
# Tiered mileage rate
if (( $(echo "$M <= 100" | bc -l) )); then
  rate=0.58
elif (( $(echo "$M <= 500" | bc -l) )); then
  rate=0.35
else
  rate=0.25
fi

# Receipts > $1 000 shift mileage rate one tier lower
if (( $(echo "$R > 1000" | bc -l) )); then
  if (( $(echo "$rate == 0.58" | bc -l) )); then
    rate=0.35
  elif (( $(echo "$rate == 0.35" | bc -l) )); then
    rate=0.25
  fi
fi

m_component=$(echo "scale=4; $rate * $M" | bc)

#—— 3. Efficiency bonus (150–220 mi per day) —————————
mpd=$(echo "scale=4; $M / $D" | bc)
if (( $(echo "$mpd >= 150 && $mpd <= 220" | bc -l) )); then
  base=$(echo "scale=4; $base * 1.10" | bc)
  m_component=$(echo "scale=4; $m_component * 1.08" | bc)
fi

#—— 4. Receipts function (diminishing returns) ———————
if (( $(echo "$R <= 600" | bc -l) )); then
  net_receipts=$(echo "scale=4; 0.8 * $R" | bc)
else
  above=$(echo "scale=4; $R - 600" | bc)
  net_receipts=$(echo "scale=4; (0.8 * 600) + (0.2 * $above)" | bc)
fi

# Tiny-receipt penalty (if receipts per day < $30)
per_day=$(echo "scale=4; $R / $D" | bc)
penalised=false
if (( $(echo "$per_day < 30" | bc -l) )); then
  base=$(echo "scale=4; $base - 40" | bc)
  penalised=true
fi

# Mileage-heavy override: remove penalty if MPD > 250
if (( $(echo "$mpd > 250" | bc -l) )) && [[ "$penalised" == "true" ]]; then
  base=$(echo "scale=4; $base + 40" | bc)
fi

# Rounding-cent windfall (+$3 if receipts end in .49 or .99)
cents=$(printf "%.2f" "$R" | awk -F'.' '{print $2}')
if [[ "$cents" == "49" || "$cents" == "99" ]]; then
  net_receipts=$(echo "scale=4; $net_receipts + 3" | bc)
fi

#—— 5. Final reimbursement ————————————————————————
raw_total=$(echo "scale=4; $base + $m_component + $net_receipts" | bc)
# round to 2 decimal places
reimbursement=$(printf "%.2f" "$raw_total")

echo "$reimbursement"