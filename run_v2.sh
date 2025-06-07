#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_v2.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    exit 1
fi

trip_duration_days=$1
miles_traveled=$2
total_receipts_amount=$3

# Variable to store the final calculation string for bc
formula_string=""

# Determine which formula to use based on total_receipts_amount
# Using awk for robust floating point comparisons

# Low Receipts (total_receipts_amount <= 200)
if (( $(awk -v val="$total_receipts_amount" 'BEGIN { exit !(val <= 200) }') )); then
    formula_string="113.74 + (0.09 * $total_receipts_amount) + (57.92 * $trip_duration_days) + (0.50 * $miles_traveled)"
# Medium Receipts (200 < total_receipts_amount <= 1000)
elif (( $(awk -v val="$total_receipts_amount" 'BEGIN { exit !(val > 200 && val <= 1000) }') )); then
    formula_string="-152.84 + (0.87 * $total_receipts_amount) + (55.77 * $trip_duration_days) + (0.53 * $miles_traveled)"
# High Receipts (total_receipts_amount > 1000)
else
    formula_string="1089.01 + (0.00 * $total_receipts_amount) + (42.04 * $trip_duration_days) + (0.37 * $miles_traveled)"
fi

# Perform the calculation using bc with a scale for precision
# Set a higher scale for intermediate calculations to maintain precision before final rounding
calculation=$(echo "scale=10; $formula_string" | bc)

# Output the result rounded to 2 decimal places
# printf is a robust way to ensure the correct formatting
printf "%.2f\\n" "$calculation"
