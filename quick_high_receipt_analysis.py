import json
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

MIN_SAMPLES_FOR_REGRESSION = 5 # Need at least num_features + 1

def load_data(file_path="public_cases.json"):
    """Loads data from the JSON file into a pandas DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    inputs = [item['input'] for item in data]
    outputs = [item['expected_output'] for item in data]
    
    df = pd.DataFrame(inputs)
    df['expected_output'] = outputs
    
    df['trip_duration_days'] = df['trip_duration_days'].astype(int)
    df['miles_traveled'] = df['miles_traveled'].astype(float)
    df['total_receipts_amount'] = df['total_receipts_amount'].astype(float)
    df['expected_output'] = df['expected_output'].astype(float)
    return df

def fit_linear_model_to_subset(df_subset, subset_name):
    """
    Fits a linear model (expected_output ~ intercept + receipts + days + miles)
    to the given DataFrame subset.
    Prints the coefficients and returns them.
    """
    print(f"\n--- Fitting Model to Subset: {subset_name} (Size: {len(df_subset)}) ---")
    if len(df_subset) < MIN_SAMPLES_FOR_REGRESSION:
        print(f"Subset too small ({len(df_subset)} samples) to fit a reliable model. Skipping.")
        return None

    X_cols = ['total_receipts_amount', 'trip_duration_days', 'miles_traveled']
    X = df_subset[X_cols].values
    # Add intercept column of ones
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    y = df_subset['expected_output'].values

    try:
        coeffs, residuals_sum_sq, rank, singular_values = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        intercept, coeff_receipts, coeff_days, coeff_miles = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        
        formula_str = (f"  Formula for '{subset_name}': expected_output = "
                       f"{intercept:.2f} (intercept) + "
                       f"{coeff_receipts:.2f} * total_receipts_amount + "
                       f"{coeff_days:.2f} * trip_duration_days + "
                       f"{coeff_miles:.2f} * miles_traveled")
        print(formula_str)
        
        if coeff_receipts < -0.01: # Using -0.01 as a threshold for "notably negative"
            print(f"  IMPORTANT FINDING: Receipt coefficient is notably NEGATIVE ({coeff_receipts:.2f}), suggesting a penalty for receipts in this segment.")
        elif coeff_receipts < 0.05 and coeff_receipts > -0.01: # Close to zero
             print(f"  NOTE: Receipt coefficient is close to ZERO ({coeff_receipts:.2f}), suggesting receipts have little to no positive impact or are ignored.")
        else:
            print(f"  NOTE: Receipt coefficient is POSITIVE ({coeff_receipts:.2f}).")
            
        return {'intercept': intercept, 'coeff_receipts': coeff_receipts, 'coeff_days': coeff_days, 'coeff_miles': coeff_miles}
                
    except np.linalg.LinAlgError as e:
        print(f"  Linear regression failed for subset '{subset_name}'. Error: {e}")
        return None

def main():
    print("--- Quick High-Receipt Analysis ---")
    
    df_all = load_data()
    
    # 1. Filter for all high receipt cases
    df_high_receipts_all = df_all[df_all['total_receipts_amount'] > 1000].copy()
    print(f"\nTotal cases with receipts > $1000: {len(df_high_receipts_all)}")
    
    # 2. Define the specific "worst high-error cases" from previous eval
    # These were cases where the previous model (segmented by total_receipts_amount) still had large errors.
    # The previous high-receipt formula was: 1089.01 + (0.00 * R) + (42.04 * D) + (0.37 * M)
    worst_cases_data = [
        {'trip_duration_days': 1, 'miles_traveled': 1082, 'total_receipts_amount': 1809.49, 'expected_output': 446.94, 'case_id': '996'},
        {'trip_duration_days': 8, 'miles_traveled': 795, 'total_receipts_amount': 1645.99, 'expected_output': 644.69, 'case_id': '684'},
        {'trip_duration_days': 8, 'miles_traveled': 482, 'total_receipts_amount': 1411.49, 'expected_output': 631.81, 'case_id': '548'},
        {'trip_duration_days': 4, 'miles_traveled': 69, 'total_receipts_amount': 2321.49, 'expected_output': 322.00, 'case_id': '152'},
        {'trip_duration_days': 4, 'miles_traveled': 286, 'total_receipts_amount': 1063.49, 'expected_output': 418.17, 'case_id': '244'}
    ]
    df_worst_high_receipt_cases = pd.DataFrame(worst_cases_data)
    
    # 3. Fit model to ALL high-receipt cases
    coeffs_all_high = fit_linear_model_to_subset(df_high_receipts_all, "All Cases with Receipts > $1000")
    
    # 4. Fit model to JUST the defined "worst high-error cases"
    coeffs_worst_cases = fit_linear_model_to_subset(df_worst_high_receipt_cases, "Specific Worst High-Receipt Error Cases")

    print("\n--- Analysis Conclusions & Recommendations ---")
    if coeffs_worst_cases and coeffs_worst_cases['coeff_receipts'] < -0.01:
        print("RECOMMENDATION: The model for the *worst* high-receipt cases shows a NEGATIVE receipt coefficient.")
        print("This strongly suggests a PENALTY for very high receipts, or a different calculation logic for them.")
        print("Consider using the formula derived from 'Specific Worst High-Receipt Error Cases' for amounts > $1000, or for a sub-segment of extremely high receipts if this group is distinct.")
        print(f"Worst cases formula: E_Out = {coeffs_worst_cases['intercept']:.2f} + ({coeffs_worst_cases['coeff_receipts']:.2f}*R) + ({coeffs_worst_cases['coeff_days']:.2f}*D) + ({coeffs_worst_cases['coeff_miles']:.2f}*M)")
    elif coeffs_all_high and coeffs_all_high['coeff_receipts'] < -0.01:
        print("RECOMMENDATION: The model for *all* high-receipt cases (> $1000) shows a NEGATIVE receipt coefficient.")
        print("This suggests a PENALTY for receipts > $1000.")
        print(f"All high-receipts formula: E_Out = {coeffs_all_high['intercept']:.2f} + ({coeffs_all_high['coeff_receipts']:.2f}*R) + ({coeffs_all_high['coeff_days']:.2f}*D) + ({coeffs_all_high['coeff_miles']:.2f}*M)")
    elif coeffs_all_high and coeffs_all_high['coeff_receipts'] < 0.05 : # Close to zero
        print("NOTE: The model for *all* high-receipt cases (> $1000) shows a receipt coefficient close to zero.")
        print("This confirms that receipts > $1000 contribute very little or are ignored, consistent with previous segmentation.")
        print(f"All high-receipts formula (confirming near-zero effect): E_Out = {coeffs_all_high['intercept']:.2f} + ({coeffs_all_high['coeff_receipts']:.2f}*R) + ({coeffs_all_high['coeff_days']:.2f}*D) + ({coeffs_all_high['coeff_miles']:.2f}*M)")
    else:
        print("No clear negative penalty for receipts found in these specific high-receipt analyses. The previous segmentation's zero coefficient might still be the best approach for >$1000 receipts, or further refinement of segments is needed.")

    print("\n--- End of Quick High-Receipt Analysis ---")

if __name__ == '__main__':
    main()
