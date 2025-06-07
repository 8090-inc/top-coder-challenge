#!/bin/bash

# run.sh - Hybrid approach combining best segmentation with ML insights.

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

# Using bc for robust floating point comparisons. bc returns 1 for true, 0 for false.
# Define ML-refined boundary for high receipts
ML_HIGH_RECEIPT_BOUNDARY="828" # Approximately sqrt(685741.84)

# Segment 1: Low Receipts (total_receipts_amount <= 200)
condition_low_receipts=$(echo "$total_receipts_amount <= 200" | bc -l)

# Segment 2: Medium Receipts (200 < total_receipts_amount <= ML_HIGH_RECEIPT_BOUNDARY)
condition_medium_receipts=$(echo "$total_receipts_amount > 200 && $total_receipts_amount <= $ML_HIGH_RECEIPT_BOUNDARY" | bc -l)

# Segment 3: High Receipts (total_receipts_amount > ML_HIGH_RECEIPT_BOUNDARY)
# This condition is met if the others are false.

if [ "$condition_low_receipts" -eq 1 ]; then
    # Formula from best manual segmentation for low receipts
    # E_Out = 113.74 + (0.09 * R) + (57.92 * D) + (0.50 * M)
    formula_string="113.74 + (0.09 * $total_receipts_amount) + (57.92 * $trip_duration_days) + (0.50 * $miles_traveled)"
elif [ "$condition_medium_receipts" -eq 1 ]; then
    # Formula from best manual segmentation for medium receipts (boundary adjusted by ML)
    # E_Out = -152.84 + (0.87 * R) + (55.77 * D) + (0.53 * M)
    formula_string="-152.84 + (0.87 * $total_receipts_amount) + (55.77 * $trip_duration_days) + (0.53 * $miles_traveled)"
else
    # Formula for High Receipts (> ML_HIGH_RECEIPT_BOUNDARY), incorporating ML insight of negative/penalizing coefficient for receipts
    # From quick_high_receipt_analysis.py for "Specific Worst High-Receipt Error Cases":
    # E_Out = 260.17 + (-0.05*R) + (39.22*D) + (0.21*M)
    formula_string="260.17 + (-0.05 * $total_receipts_amount) + (39.22 * $trip_duration_days) + (0.21 * $miles_traveled)"
fi

# Perform the calculation using bc with a scale for precision
# Set a higher scale for intermediate calculations to maintain precision before final rounding
calculation=$(echo "scale=10; $formula_string" | bc)

# Output the result rounded to 2 decimal places
# printf is a robust way to ensure the correct formatting
printf "%.2f\\n" "$calculation"
