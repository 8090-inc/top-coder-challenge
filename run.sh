#!/bin/bash

# run.sh
# This script serves as an interface for the eval.sh harness.
# It takes three command-line arguments: days, miles, and receipts,
# and then calls the Python function in final_function.py to calculate
# the reimbursement, outputting the result to stdout.

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
    exit 1
fi

DAYS=$1
MILES=$2
RECEIPTS=$3

# Call the Python function from final_function.py
PYTHON_SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null
then
    PYTHON_CMD="python"
fi

# Python inline script to call the function and print the result
OUTPUT=$("$PYTHON_CMD" -c "
import sys
sys.path.append('$PYTHON_SCRIPT_DIR')
from final_function import calculate_reimbursement

try:
    days_val = int('$DAYS')
    miles_val = float('$MILES') # Ensure miles is float for the function
    receipts_val = float('$RECEIPTS')
    result = calculate_reimbursement(days_val, miles_val, receipts_val)
    print(f\"{result:.2f}\")
except ValueError as e:
    # print(f\"Error in Python script: ValueError converting arguments: {e}\", file=sys.stderr)
    sys.exit(2)
except ImportError as e:
    # print(f\"Error in Python script: ImportError: {e}. Ensure final_function.py is in the same directory or PYTHONPATH.\", file=sys.stderr)
    sys.exit(3)
except Exception as e:
    # print(f\"Error in Python script: An unexpected error occurred: {e}\", file=sys.stderr)
    sys.exit(4)
")

PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    exit $PYTHON_EXIT_CODE
else
    echo "$OUTPUT"
fi
