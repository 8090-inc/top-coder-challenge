#!/usr/bin/env python3
"""
Command‑line prediction entry point.

Usage:
    ./predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>

This script is designed to be called by `run.sh` and by the instructor's
evaluation scripts. It simply parses the three inputs, loads the production
predictor, and prints a single reimbursement value.
"""
import sys

from pathlib import Path

from 07_production_pipeline import ReimbursementPredictor


def main(argv) -> None:
    if len(argv) != 4:
        sys.stderr.write(
            "Usage: predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>\n"
        )
        # Still print something numeric so automated scripts do not crash
        print("0.00")
        return

    try:
        days = float(argv[1])
        miles = float(argv[2])
        receipts = float(argv[3])
    except ValueError:
        sys.stderr.write("All three arguments must be numeric.\n")
        print("0.00")
        return

    predictor = ReimbursementPredictor()
    value = predictor.predict_one(days, miles, receipts)
    # Ensure exactly two decimals
    print(f"{value:.2f}")


if __name__ == "__main__":
    main(sys.argv)
