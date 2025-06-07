#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
  exit 1
fi

TRIP_DURATION_DAYS="$1"
MILES_TRAVELED="$2"
TOTAL_RECEIPTS_AMOUNT="$3"

# Validate input types (basic validation, Python script will also validate)
if ! [[ "$TRIP_DURATION_DAYS" =~ ^[0-9]+$ ]]; then
    echo "Error: trip_duration_days must be an integer." >&2
    exit 1
fi
# Miles and receipts can be float, more complex regex, rely on python for float conversion.

# Check if reimbursement_engine.py exists
if [ ! -f "reimbursement_engine.py" ]; then
  echo "Error: reimbursement_engine.py not found in the current directory." >&2
  exit 1
fi

# Check if temp_runner.py exists
if [ ! -f "temp_runner.py" ]; then
  echo "Error: temp_runner.py not found in the current directory." >&2
  exit 1
fi

# Set PYTHONPATH to include the current directory for the engine import
export PYTHONPATH=$(pwd):$PYTHONPATH

# Execute the Python script that handles the processing
python3 temp_runner.py "$TRIP_DURATION_DAYS" "$MILES_TRAVELED" "$TOTAL_RECEIPTS_AMOUNT"
