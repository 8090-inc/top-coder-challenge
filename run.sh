#!/bin/bash

# Black Box Challenge - Legacy Reimbursement Calculator v0.5
# This script calculates reimbursement using our cluster-based model
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Run the Python implementation
python3 calculate_reimbursement.py "$1" "$2" "$3" 2>/dev/null | grep "Expected Reimbursement:" | awk '{print $3}' | tr -d '$' 