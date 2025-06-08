import json
import pandas as pd
import numpy as np

# Assuming reimbursement_calculator.py is in the same directory and contains the v4 logic
from reimbursement_calculator import calculate_mileage_reimbursement, calculate_receipt_contribution

def load_data(filepath):
    """Loads data from a JSON file and normalizes the nested 'input' fields."""
    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        df_initial = pd.DataFrame(raw_data)
        if 'input' not in df_initial.columns:
            print("Error: 'input' column not found.")
            return None
        input_data_normalized = pd.json_normalize(df_initial['input'])
        if 'expected_output' not in df_initial.columns:
            print("Error: 'expected_output' column not found.")
            # In a real scenario, might return df with only input cols or handle as error
            return input_data_normalized
        df_final = pd.concat([input_data_normalized, df_initial[['expected_output']]], axis=1)
        return df_final
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def calibrate_new_per_diems():
    data_filepath = 'public_cases.json'
    df = load_data(data_filepath)

    if df is None:
        print("Failed to load data. Exiting calibration.")
        return {}

    # Convert relevant columns to numeric, coercing errors
    df['trip_duration_days'] = pd.to_numeric(df['trip_duration_days'], errors='coerce')
    df['miles_traveled'] = pd.to_numeric(df['miles_traveled'], errors='coerce')
    df['total_receipts_amount'] = pd.to_numeric(df['total_receipts_amount'], errors='coerce')
    df['expected_output'] = pd.to_numeric(df['expected_output'], errors='coerce')

    # Drop rows where conversion failed for essential columns
    df.dropna(subset=['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output'], inplace=True)
    df['trip_duration_days'] = df['trip_duration_days'].astype(int)


    new_per_diem_lookup = {}
    print("Calculating new per diem values:")

    for days_val in range(1, 15): # For durations 1 to 14
        duration_cases = df[df['trip_duration_days'] == days_val]
        if duration_cases.empty:
            print(f"No cases found for {days_val} days. Will need placeholder or interpolation.")
            continue

        base_amounts_for_duration = []
        for index, row in duration_cases.iterrows():
            # Use functions from the latest reimbursement_calculator.py (v4)
            # calculate_mileage_reimbursement now needs 'days'
            mileage_comp = calculate_mileage_reimbursement(row['miles_traveled'], row['trip_duration_days'])
            receipt_comp = calculate_receipt_contribution(row['total_receipts_amount'])

            base_amount = row['expected_output'] - (mileage_comp + receipt_comp)
            base_amounts_for_duration.append(base_amount)

        avg_base_amount = np.mean(base_amounts_for_duration)
        new_per_diem_lookup[days_val] = round(avg_base_amount, 2)
        print(f"  {days_val} days: Avg Base Amount = {avg_base_amount:.2f}")

    print("\nNew PER_DIEM_LOOKUP dictionary:")
    print("{")
    for day, amount in new_per_diem_lookup.items():
        print(f"    {day}: {amount},")
    print("}")

    # Suggesting a fallback based on trend or a new placeholder
    # For now, let's just note the default from the problem description:
    print("\nFallback for days > 14: days * 50.0 (as per problem spec)")

    return new_per_diem_lookup

if __name__ == "__main__":
    calibrate_new_per_diems()
