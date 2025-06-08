import json
import pandas as pd
import numpy as np

# Assuming final_function.py contains the v8.7.1 logic
from final_function import calculate_reimbursement

def load_data(filepath):
    """Loads data from a JSON file and normalizes the nested 'input' fields."""
    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        df_initial = pd.DataFrame(raw_data)
        # Ensure 'input' column exists before trying to normalize
        if 'input' not in df_initial.columns:
            print(f"Error: 'input' column not found in {filepath}")
            return None
        input_data_normalized = pd.json_normalize(df_initial['input'])
        # Ensure 'expected_output' column exists
        if 'expected_output' not in df_initial.columns:
            print(f"Error: 'expected_output' column not found in {filepath}")
            return None # Or handle appropriately
        df_final = pd.concat([input_data_normalized, df_initial[['expected_output']]], axis=1)
        return df_final
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def find_the_top_error_case_v8_7_1():
    data_filepath = 'public_cases.json'
    df = load_data(data_filepath)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Ensure correct data types for calculation
    try:
        df['trip_duration_days'] = pd.to_numeric(df['trip_duration_days'], errors='coerce').fillna(0).astype(int)
        df['miles_traveled'] = pd.to_numeric(df['miles_traveled'], errors='coerce').fillna(0.0)
        df['total_receipts_amount'] = pd.to_numeric(df['total_receipts_amount'], errors='coerce').fillna(0.0)
        df['expected_output'] = pd.to_numeric(df['expected_output'], errors='coerce').fillna(0.0)
    except KeyError as e:
        print(f"Error: A required column is missing from public_cases.json: {e}")
        return

    df.dropna(subset=['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output'], inplace=True)


    max_abs_error = -1.0
    top_error_case_details = None

    print("Analyzing cases to find top error using v8.7.1 logic...")
    for index, row in df.iterrows():
        days_val = int(row['trip_duration_days'])
        miles_val = float(row['miles_traveled'])
        receipts_val = float(row['total_receipts_amount'])
        expected_val = float(row['expected_output'])

        calculated_val = calculate_reimbursement(days_val, miles_val, receipts_val)

        current_error = expected_val - calculated_val # Signed error
        abs_error = abs(current_error)

        if abs_error > max_abs_error:
            max_abs_error = abs_error
            top_error_case_details = {
                "index": index,
                "days": days_val,
                "miles": miles_val,
                "receipts": receipts_val,
                "expected_output": expected_val,
                "calculated_output (v8.7.1)": calculated_val,
                "signed_error (expected - calculated)": current_error,
                "absolute_error": abs_error
            }

    if top_error_case_details:
        print("\n--- Top Error Case Details (v8.7.1 logic) ---")
        for key, value in top_error_case_details.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("No cases processed or error in finding top error case.")

if __name__ == "__main__":
    find_the_top_error_case_v8_7_1()
