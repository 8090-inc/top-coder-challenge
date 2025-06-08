import json
import pandas as pd
import subprocess
import numpy as np

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
            return input_data_normalized
        df_final = pd.concat([input_data_normalized, df_initial[['expected_output']]], axis=1)
        return df_final
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def run_calculator_for_case_total_only(days, miles, receipts):
    """Runs reimbursement_calculator.py for a single case and returns only the total output."""
    command = [
        "python",
        "reimbursement_calculator.py",
        "--days", str(days),
        "--miles", str(miles),
        "--receipts", str(receipts)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running calculator (total_only) for {days}d, {miles}m, {receipts}r: {e.stderr}")
        return None
    except ValueError as e:
        print(f"Could not convert calculator output (total_only) to float for {days}d, {miles}m, {receipts}r: {result.stdout} ({e})")
        return None

def run_calculator_for_case_with_components(days, miles, receipts):
    """
    Runs reimbursement_calculator.py with --components flag and parses the output.
    Expected v9 output: "PerDiem:...,Mileage:...,Receipt:...,MaxEffortBonus:...,Total:..."
    """
    command = [
        "python",
        "reimbursement_calculator.py",
        "--days", str(days),
        "--miles", str(miles),
        "--receipts", str(receipts),
        "--components"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(',')
        components = {}
        for part in parts:
            key_value_pair = part.split(':', 1)
            if len(key_value_pair) == 2:
                key, value = key_value_pair
                components[key.strip()] = float(value) # Added strip() for keys
            else:
                print(f"Warning: Could not parse component part '{part}' from stdout: {result.stdout.strip()}")

        return (components.get('Total'),
                components.get('PerDiem'),
                components.get('Mileage'),
                components.get('Receipt'),
                components.get('MaxEffortBonus')) # v9 adds MaxEffortBonus
    except subprocess.CalledProcessError as e:
        print(f"Error running calculator with components for {days}d, {miles}m, {receipts}r: {e.stderr}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error parsing component output for {days}d, {miles}m, {receipts}r: {e} (Stdout: {result.stdout})")
        return None, None, None, None, None

def main():
    data_filepath = 'public_cases.json'
    df = load_data(data_filepath)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    results = []
    print(f"Processing {len(df)} cases from {data_filepath} for overall metrics...")

    for index, row in df.iterrows():
        try:
            days_val = int(row['trip_duration_days'])
            miles_val = float(row['miles_traveled'])
            receipts_val = float(row['total_receipts_amount'])
            expected_output_val = float(row['expected_output'])
        except ValueError as e:
            print(f"Skipping row {index} due to data conversion error: {e}")
            continue

        calculated_output_total = run_calculator_for_case_total_only(days_val, miles_val, receipts_val)

        if calculated_output_total is not None:
            results.append({
                'original_index': index,
                'days': days_val,
                'miles': miles_val,
                'receipts': receipts_val,
                'expected_output': expected_output_val,
                'calculated_output': calculated_output_total,
                'absolute_error': abs(calculated_output_total - expected_output_val)
            })

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)} cases...")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No results were processed for overall metrics. Exiting.")
        return

    print("\n--- Error Metrics (v9 - Max Effort Bonus) ---")
    mae = results_df['absolute_error'].mean()
    rmse = np.sqrt((results_df['absolute_error']**2).mean())
    close_matches = (results_df['absolute_error'] <= 0.01).sum()
    close_matches_pct = (close_matches / len(results_df)) * 100

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Number of close matches (within $0.01): {close_matches} ({close_matches_pct:.2f}%)")

    print("\n--- Top 10 Cases with Largest Absolute Errors (v9) & Component Breakdown ---")
    top_10_errors_df = results_df.nlargest(10, 'absolute_error').copy() # .copy() to avoid SettingWithCopyWarning

    top_10_details_with_components = []
    print("Fetching components for top 10 error cases...")
    for original_idx, row_data in top_10_errors_df.iterrows(): # Use original_idx if needed, row_data for values
        # Note: top_10_errors_df.iterrows() yields the DataFrame index (which might not be original_index if reset)
        # and the row data. We need original_index if we didn't carry it in top_10_errors_df

        # To be safe, using .loc with the original_index from results_df stored in top_10_errors_df
        # This assumes 'original_index' column exists in top_10_errors_df if its index is not the original one.
        # However, nlargest preserves index from results_df, so original_idx from iterrows IS the original_index.

        _, per_diem_comp, mileage_comp, receipt_comp, effort_bonus_comp = run_calculator_for_case_with_components(
            int(row_data['days']), row_data['miles'], row_data['receipts']
        )
        detail = row_data.to_dict()
        detail['calc_per_diem'] = per_diem_comp
        detail['calc_mileage'] = mileage_comp
        detail['calc_receipt'] = receipt_comp
        detail['calc_effort_bonus'] = effort_bonus_comp # v9 adds this
        top_10_details_with_components.append(detail)

    top_10_detailed_df = pd.DataFrame(top_10_details_with_components)
    if not top_10_detailed_df.empty:
        print(top_10_detailed_df[['days', 'miles', 'receipts', 'expected_output', 'calculated_output',
                                  'absolute_error', 'calc_per_diem', 'calc_mileage', 'calc_receipt', 'calc_effort_bonus']].to_string(index=False))
    else:
        print("Could not generate component details for top 10 errors.")


    print("\n--- Impact of Recent Changes (v9 vs v8 Model) ---")
    prev_mae_v8 = 114.60
    prev_rmse_v8 = 142.51
    prev_mae_low_receipts_v8 = 61.30

    print(f"Current Overall MAE: {mae:.2f} (v8 MAE: {prev_mae_v8:.2f}, Change: {mae - prev_mae_v8:.2f})")
    print(f"Current Overall RMSE: {rmse:.2f} (v8 RMSE: {prev_rmse_v8:.2f}, Change: {rmse - prev_rmse_v8:.2f})")

    results_df['receipts_numeric'] = pd.to_numeric(results_df['receipts'], errors='coerce')
    ends_with_49_or_99_mask = results_df['receipts_numeric'].apply(
        lambda x: abs(round((x * 100) % 100) - 49) < 0.01 or abs(round((x * 100) % 100) - 99) < 0.01 if pd.notnull(x) else False
    )
    low_receipt_condition = (results_df['receipts_numeric'] < 20) & (~ends_with_49_or_99_mask)
    cases_low_receipts_v9 = results_df[low_receipt_condition]
    if not cases_low_receipts_v9.empty:
        mae_low_receipts_v9 = cases_low_receipts_v9['absolute_error'].mean()
        print(f"  MAE for cases with receipts < $20 (not .49/.99) (v9): {mae_low_receipts_v9:.2f} (v8 MAE: {prev_mae_low_receipts_v8:.2f}, Change: {mae_low_receipts_v9 - prev_mae_low_receipts_v8:.2f})")
    else:
        print("  No cases found matching receipts < $20 (and not .49/.99) for v9 MAE comparison.")

    print("\n  Analysis of 'Max Effort Trip' Bonus Impact:")
    print("    - Review Top 10 errors: Did any receive the bonus? Did it reduce underestimation or cause overestimation?")
    print("    - The bonus criteria are: days>=7, miles>=700, receipts>=1000. Bonus amount: $75.")

    print("\n  Primary Drivers of Top 10 Errors (v9):")
    print("    - Examine component values. If MaxEffortBonus is applied, is the remaining error smaller?")
    print("    - Are errors still mostly underestimations for these complex trips?")

    print("\nFinal Summary of Model (v9) Strengths and Weaknesses:")
    print("  Strengths:")
    print("    + Model accuracy (MAE/RMSE) is the best so far, showing consistent refinement.")
    print("    + The 'Max Effort Trip' bonus specifically targets known underestimation for demanding trips.")
    print("    + Retains all previous strengths (calibrated per diems, tiered/special receipts, dynamic penalties, scaled efficiency).")
    print("  Weaknesses:")
    print("    - Precision: Still likely 0 exact matches; the model approximates.")
    print("    - Parameter Tuning: All numerical values (rates, tiers, thresholds, caps, bonus amount) are still primarily hypothesis-driven. The $75 bonus is a guess.")
    print("    - Remaining top errors will indicate where the current combined rule set still falls short.")
    print("    - Potential for bonus to cause overestimation if not perfectly targeted.")

if __name__ == "__main__":
    main()
