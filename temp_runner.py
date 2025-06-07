import json
import sys
from reimbursement_engine import ReimbursementEngine # Assuming the class is named ReimbursementEngine

def main():
    if len(sys.argv) != 4:
        print("Usage: python temp_runner.py <trip_duration_days> <miles_traveled> <total_receipts_amount>", file=sys.stderr)
        sys.exit(1)

    try:
        trip_duration_days = int(sys.argv[1])
        miles_traveled = float(sys.argv[2]) # Miles can be float based on some public cases
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        print("Error: Invalid input types. Duration must be int, miles and receipts must be numbers.", file=sys.stderr)
        print("0.00") # Default error output
        sys.exit(1)

    input_case_dict = {
        "trip_duration_days": trip_duration_days,
        "miles_traveled": miles_traveled,
        "total_receipts_amount": total_receipts_amount
    }

    engine = ReimbursementEngine() # Initialize once with public_cases.json (default)

    try:
        result = engine.calculate_reimbursement(input_case_dict)
        print(result)
    except Exception as e:
        print(f"Error processing case {input_case_dict}: {e}", file=sys.stderr)
        print("0.00") # Default error output

if __name__ == "__main__":
    main()
