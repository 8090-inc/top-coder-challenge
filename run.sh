#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    exit 1
fi

trip_duration_days=$1
miles_traveled=$2
total_receipts_amount=$3

# Variable to store the final calculation string for bc
formula_string=""

# Determine which formula to use based on total_receipts_amount
# Using bc for robust floating point comparisons. bc returns 1 for true, 0 for false.

# Low Receipts (total_receipts_amount <= 200)
condition_low_receipts=$(echo "$total_receipts_amount <= 200" | bc -l)

# Medium Receipts (200 < total_receipts_amount <= 1000)
condition_medium_receipts=$(echo "$total_receipts_amount > 200 && $total_receipts_amount <= 1000" | bc -l)

if [ "$condition_low_receipts" -eq 1 ]; then
    # Low Receipts (<= $200)
    formula_string="113.74 + (0.09 * $total_receipts_amount) + (57.92 * $trip_duration_days) + (0.50 * $miles_traveled)"
elif [ "$condition_medium_receipts" -eq 1 ]; then
    # Medium Receipts ($201-$1000)
    formula_string="-152.84 + (0.87 * $total_receipts_amount) + (55.77 * $trip_duration_days) + (0.53 * $miles_traveled)"
else
    # High Receipts (> $1000) - Receipts have 0.00 coefficient (effectively ignored)
    formula_string="1089.01 + (0.00 * $total_receipts_amount) + (42.04 * $trip_duration_days) + (0.37 * $miles_traveled)"
fi

# Perform the calculation using bc with a scale for precision
# Set a higher scale for intermediate calculations to maintain precision before final rounding
calculation=$(echo "scale=10; $formula_string" | bc)

# Output the result rounded to 2 decimal places
# printf is a robust way to ensure the correct formatting
printf "%.2f\\n" "$calculation"
