#!/bin/bash

# Black Box Challenge - Legacy Reimbursement Calculator v5
# This script calculates reimbursement using our v5 practical ensemble model
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Run the Python implementation and extract just the number
python3 calculate_reimbursement.py "$1" "$2" "$3" 2>/dev/null | grep "Expected Reimbursement:" | awk -F'$' '{print $2}' 