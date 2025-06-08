#!/usr/bin/env bash

# Legacy Reimbursement System — PURE-PYTHON inference
# Uses the auto-generated decision-tree scorer in src/generated_tree_calculator.py
# No external Python packages required at run-time.
#
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

set -euo pipefail

TMP_PY=$(mktemp)
cat > "$TMP_PY" <<'ENDPY'
import sys
sys.path.append('src')
from generated_tree_calculator import calculate_reimbursement_tree

if len(sys.argv) != 4:
    print('Expected 3 arguments: days miles receipts', file=sys.stderr)
    sys.exit(1)

try:
    days = float(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
except ValueError as exc:
    print(f'Invalid input: {exc}', file=sys.stderr)
    sys.exit(1)

result = calculate_reimbursement_tree(days, miles, receipts)
print(f"{result:.2f}")
ENDPY

python3 "$TMP_PY" "$@"
rm -f "$TMP_PY" 