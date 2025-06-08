#!/bin/bash

# Black Box Challenge - V6 Model Runner
# This script calculates reimbursement using our V6 XGBoost + LightGBM ensemble model
# Usage: ./run_v6.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Run the V6 Python implementation and output just the number
python3 run_v6.py "$1" "$2" "$3" 2>/dev/null 