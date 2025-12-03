#!/bin/bash
# Thin wrapper around predict.py so that:
#   ./run.sh DAYS MILES RECEIPTS
# prints a single reimbursement value.
#
# This should be compatible with eval.sh / generate_results.sh.

set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
  echo "0.00"
  exit 0
fi

days="$1"
miles="$2"
receipts="$3"

./predict.py "$days" "$miles" "$receipts"
