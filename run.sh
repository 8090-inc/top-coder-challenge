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

#—— 1. Base calculation (decreases significantly with trip length) ———
# Pattern: 1-day = ~$870/day, but drops to ~$150/day for 10+ days
# Appears to be a fixed base amount divided by trip length with scaling
if (( $(echo "$D == 1" | bc -l) )); then
  base=$(echo "scale=4; 120" | bc)
elif (( $(echo "$D <= 3" | bc -l) )); then
  base=$(echo "scale=4; 60 * $D" | bc)
elif (( $(echo "$D <= 7" | bc -l) )); then
  base=$(echo "scale=4; 50 * $D" | bc)
else
  base=$(echo "scale=4; 45 * $D" | bc)
fi

#—— 2. Mileage component (simple rate structure) ——————————————————
# From analysis: around $0.3-0.4/mile for most cases
m_component=$(echo "scale=4; $M * 0.32" | bc)

#—— 3. Receipt processing (strong diminishing returns) ————————————
# Low receipts get high multipliers, high receipts get heavily penalized
if (( $(echo "$R <= 50" | bc -l) )); then
  # Low receipts get boosted significantly  
  net_receipts=$(echo "scale=4; $R * 2.5" | bc)
elif (( $(echo "$R <= 200" | bc -l) )); then
  # Medium receipts get good treatment
  net_receipts=$(echo "scale=4; $R * 1.8" | bc)
elif (( $(echo "$R <= 600" | bc -l) )); then
  # Higher receipts start getting diminishing returns
  net_receipts=$(echo "scale=4; $R * 1.0" | bc)
else
  # Very high receipts get penalized heavily
  base_600=$(echo "scale=4; 600 * 1.0" | bc)
  excess=$(echo "scale=4; $R - 600" | bc)
  net_receipts=$(echo "scale=4; $base_600 + ($excess * 0.3)" | bc)
fi

#—— 4. Trip length scaling bonus/penalty ——————————————————————————
# Short trips get efficiency bonuses, long trips get penalties
length_adjustment=0
if (( $(echo "$D == 1" | bc -l) )); then
  # 1-day trips get massive bonuses (observed in data)
  length_adjustment=$(echo "scale=4; ($base + $m_component) * 4" | bc)
elif (( $(echo "$D == 2" | bc -l) )); then
  # 2-day trips get significant bonuses
  length_adjustment=$(echo "scale=4; ($base + $m_component) * 1.5" | bc)
elif (( $(echo "$D == 3" | bc -l) )); then
  # 3-day trips get moderate bonuses
  length_adjustment=$(echo "scale=4; ($base + $m_component) * 0.8" | bc)
elif (( $(echo "$D >= 5 && $D <= 7" | bc -l) )); then
  # 5-7 day trips are the "normal" range
  length_adjustment=0
else
  # Longer trips get penalties
  length_adjustment=$(echo "scale=4; ($base + $m_component) * -0.2" | bc)
fi

#—— 5. Windfall rounding (.49/.99 cents) ——————————————————————————
cents=$(printf "%.2f" "$R" | awk -F'.' '{print $2}')
windfall=0
if [[ "$cents" == "49" || "$cents" == "99" ]]; then
  windfall=$(echo "scale=4; 3" | bc)
fi

#—— 6. Final calculation ————————————————————————————————————————
raw_total=$(echo "scale=4; $base + $m_component + $net_receipts + $length_adjustment + $windfall" | bc)

# Ensure non-negative result
if (( $(echo "$raw_total < 0" | bc -l) )); then
  raw_total=0
fi

# Round to 2 decimal places
reimbursement=$(printf "%.2f" "$raw_total")
echo "$reimbursement" 