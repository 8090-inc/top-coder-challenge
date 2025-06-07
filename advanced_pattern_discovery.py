import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore', category=UserWarning, message='FigureCanvasAgg is non-interactive')
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Helper Functions ---
def load_data(file_path="public_cases.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    inputs = [item['input'] for item in data]
    outputs = [item['expected_output'] for item in data]
    df = pd.DataFrame(inputs)
    df['expected_output'] = outputs
    # Ensure correct data types
    df['trip_duration_days'] = df['trip_duration_days'].astype(int)
    df['miles_traveled'] = df['miles_traveled'].astype(float)
    df['total_receipts_amount'] = df['total_receipts_amount'].astype(float)
    df['expected_output'] = df['expected_output'].astype(float)
    return df

def create_derived_features(df_in):
    df = df_in.copy()
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days'].replace(0, np.nan)
    df['receipt_amount_per_day'] = df['total_receipts_amount'] / df['trip_duration_days'].replace(0, np.nan)
    # Linear model coefficients from previous analysis
    df['predicted_linear'] = 266.71 + \
                             (0.38 * df['total_receipts_amount']) + \
                             (50.05 * df['trip_duration_days']) + \
                             (0.45 * df['miles_traveled'])
    df['residual_linear'] = df['expected_output'] - df['predicted_linear']
    return df

# --- Analysis Functions ---

def analyze_high_error_cases(df, findings):
    print("\n--- 1. Analysis of High-Error Cases (from Linear Model) ---")
    # Cases where linear model significantly over-predicted (large negative residual)
    over_predictions = df[df['residual_linear'] < -100].sort_values(by='residual_linear')
    findings.append(f"Found {len(over_predictions)} cases where linear model over-predicted by >$100.")
    print(f"Top 5 over-prediction cases (linear model was too high):\n{over_predictions[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'predicted_linear', 'residual_linear']].head()}")

    # Example Case 152: 4 days, 69 miles, $2321.49 receipts. Expected: $322.00, Predicted: $1380.13, Residual: -$1058.13
    # Example Case 996: 1 day, 1082 miles, $1809.49 receipts. Expected: $446.94, Predicted: $1491.27, Residual: -$1044.33
    # These often have very high total_receipts_amount or receipt_amount_per_day.
    
    if not over_predictions.empty:
        avg_receipts_overpred = over_predictions['total_receipts_amount'].mean()
        avg_rapd_overpred = over_predictions['receipt_amount_per_day'].mean()
        findings.append(f"Over-predicted cases: Avg total receipts ${avg_receipts_overpred:.2f}, Avg RAPD ${avg_rapd_overpred:.2f}. This suggests issues with high receipts.")
        print(f"For these over-predicted cases, average total_receipts_amount is ${avg_receipts_overpred:.2f} and receipt_amount_per_day is ${avg_rapd_overpred:.2f}.")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_receipts_amount', y='residual_linear', data=df, hue='trip_duration_days', alpha=0.7, palette='viridis')
    plt.title('Linear Model Residual vs. Total Receipt Amount')
    plt.xlabel('Total Receipt Amount')
    plt.ylabel('Residual (Expected - Predicted)')
    plt.axhline(0, color='red', linestyle='--')
    plt.savefig('residual_vs_receipts.png')
    plt.close()
    findings.append("Saved 'residual_vs_receipts.png'. Check for patterns, especially large negative residuals at high receipt amounts.")
    print("Saved 'residual_vs_receipts.png'. Large negative residuals at high receipt amounts indicate the linear model overpays for high receipts.")

def investigate_receipt_caps_penalties(df, findings):
    print("\n--- 2. Investigating Receipt Caps/Penalties ---")
    
    # Hypothesis: Reimbursement from receipts is capped or has diminishing returns.
    # Let's estimate non-receipt part of linear model: base_others = 266.71 + 50.05 * days + 0.45 * miles
    df['linear_base_plus_mileage'] = 266.71 + (50.05 * df['trip_duration_days']) + (0.45 * df['miles_traveled'])
    df['observed_receipt_contribution'] = df['expected_output'] - df['linear_base_plus_mileage']
    df['predicted_linear_receipt_contribution'] = 0.38 * df['total_receipts_amount']

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='total_receipts_amount', y='observed_receipt_contribution', data=df, label='Observed Receipt Contrib.', alpha=0.5)
    sns.scatterplot(x='total_receipts_amount', y='predicted_linear_receipt_contribution', data=df, label='Linear Model Receipt Contrib. (0.38*R)', alpha=0.5, color='red')
    plt.title('Observed vs. Linear Model Receipt Contribution')
    plt.xlabel('Total Receipt Amount')
    plt.ylabel('Estimated Reimbursement from Receipts')
    plt.legend()
    plt.ylim(-500, 1500) # Zoom in
    plt.xlim(0, df['total_receipts_amount'].quantile(0.98)) # Zoom in on x
    plt.savefig('receipt_contribution_analysis.png')
    plt.close()
    findings.append("Saved 'receipt_contribution_analysis.png'. If observed points flatten while linear (red) points rise, it indicates a cap or diminishing returns on receipts.")
    print("Saved 'receipt_contribution_analysis.png'.")

    # Look for a cap on total_receipts_amount contribution.
    # The plot suggests that for total_receipts_amount > ~$1000-1500, the observed_receipt_contribution doesn't increase much beyond ~$400-600.
    # Let's test a cap on the *contribution* of receipts.
    # If 0.38 * total_receipts_amount > MAX_RECEIPT_CONTRIBUTION_ALLOWED, then use MAX_RECEIPT_CONTRIBUTION_ALLOWED.
    MAX_RECEIPT_CONTRIBUTION_GUESS = 500 # Eyeballing from plot
    
    df['capped_linear_receipt_contribution'] = np.minimum(df['predicted_linear_receipt_contribution'], MAX_RECEIPT_CONTRIBUTION_GUESS)
    df['predicted_with_receipt_cap'] = df['linear_base_plus_mileage'] + df['capped_linear_receipt_contribution']
    df['residual_with_receipt_cap'] = df['expected_output'] - df['predicted_with_receipt_cap']

    original_avg_abs_error = df['residual_linear'].abs().mean()
    capped_avg_abs_error = df['residual_with_receipt_cap'].abs().mean()

    findings.append(f"Receipt Cap Test: Capping linear receipt contribution at ${MAX_RECEIPT_CONTRIBUTION_GUESS} reduces avg abs error from ${original_avg_abs_error:.2f} to ${capped_avg_abs_error:.2f}.")
    print(f"With a receipt contribution cap of ${MAX_RECEIPT_CONTRIBUTION_GUESS}, average absolute error changes from ${original_avg_abs_error:.2f} to ${capped_avg_abs_error:.2f}.")
    if capped_avg_abs_error < original_avg_abs_error:
        findings.append(f"CONFIRMED: A cap on receipt contribution (around ${MAX_RECEIPT_CONTRIBUTION_GUESS}) improves the model. The 0.38 multiplier applies up to a point.")
        print("This suggests a cap on how much receipts can contribute is a valid rule.")
    
    # Alternative: Cap on receipt_amount_per_day
    RAPD_CAP_GUESS = 150 # dollars per day
    df['capped_rapd_receipt_total'] = np.minimum(df['total_receipts_amount'], RAPD_CAP_GUESS * df['trip_duration_days'])
    df['predicted_with_rapd_cap'] = 266.71 + \
                                 (0.38 * df['capped_rapd_receipt_total']) + \
                                 (50.05 * df['trip_duration_days']) + \
                                 (0.45 * df['miles_traveled'])
    df['residual_with_rapd_cap'] = df['expected_output'] - df['predicted_with_rapd_cap']
    rapd_capped_avg_abs_error = df['residual_with_rapd_cap'].abs().mean()
    findings.append(f"RAPD Cap Test: Capping RAPD at ${RAPD_CAP_GUESS}/day (effective total receipts capped) changes avg abs error to ${rapd_capped_avg_abs_error:.2f}.")
    print(f"With a receipt_amount_per_day cap of ${RAPD_CAP_GUESS}/day, average absolute error changes to ${rapd_capped_avg_abs_error:.2f}.")
    
    # Choose the better cap for now for further analysis
    if capped_avg_abs_error < rapd_capped_avg_abs_error and capped_avg_abs_error < original_avg_abs_error:
        best_receipt_cap_strategy = "Total Contribution Cap"
        best_receipt_cap_value = MAX_RECEIPT_CONTRIBUTION_GUESS
        df['current_best_receipt_contrib'] = df['capped_linear_receipt_contribution']
        df['current_best_predicted'] = df['predicted_with_receipt_cap']
        df['current_best_residual'] = df['residual_with_receipt_cap']
    elif rapd_capped_avg_abs_error < original_avg_abs_error:
        best_receipt_cap_strategy = "RAPD Cap"
        best_receipt_cap_value = RAPD_CAP_GUESS
        df['current_best_receipt_contrib'] = 0.38 * df['capped_rapd_receipt_total']
        df['current_best_predicted'] = df['predicted_with_rapd_cap']
        df['current_best_residual'] = df['residual_with_rapd_cap']
    else: # No cap was better or only one was tried and wasn't better
        best_receipt_cap_strategy = "None (Linear)"
        best_receipt_cap_value = None
        df['current_best_receipt_contrib'] = df['predicted_linear_receipt_contribution']
        df['current_best_predicted'] = df['predicted_linear']
        df['current_best_residual'] = df['residual_linear']
        
    findings.append(f"Selected Receipt Strategy for now: {best_receipt_cap_strategy} (Value: {best_receipt_cap_value}).")
    print(f"Selected Receipt Strategy for now: {best_receipt_cap_strategy} (Value: {best_receipt_cap_value}).")


def analyze_one_day_trips_deep_dive(df, findings):
    print("\n--- 3. Deep Dive into 1-Day Trips ---")
    one_day_trips = df[df['trip_duration_days'] == 1].copy()
    if one_day_trips.empty:
        findings.append("1-Day Trips: No 1-day trips found for deep dive.")
        print("No 1-day trips found.")
        return

    # Using current best receipt contribution and base mileage rate (0.45)
    # Fixed_Component_1_Day = expected_output - current_best_receipt_contrib_for_1day - (0.45 * miles)
    one_day_trips['mileage_contrib_estimate'] = 0.45 * one_day_trips['miles_traveled']
    # Need current_best_receipt_contrib for these 1-day trips from the main df
    one_day_trips_merged_receipt_contrib = pd.merge(one_day_trips, df[['current_best_receipt_contrib']], left_index=True, right_index=True, suffixes=('', '_maindf'))
    
    one_day_trips['estimated_fixed_component_1_day'] = one_day_trips_merged_receipt_contrib['expected_output'] - \
                                                       one_day_trips_merged_receipt_contrib['current_best_receipt_contrib'] - \
                                                       one_day_trips_merged_receipt_contrib['mileage_contrib_estimate']

    avg_fixed_comp_1_day = one_day_trips['estimated_fixed_component_1_day'].mean()
    median_fixed_comp_1_day = one_day_trips['estimated_fixed_component_1_day'].median()
    std_fixed_comp_1_day = one_day_trips['estimated_fixed_component_1_day'].std()

    findings.append(f"1-Day Trips: Estimated Fixed Component (Base + 1 Day Per Diem) - Mean: ${avg_fixed_comp_1_day:.2f}, Median: ${median_fixed_comp_1_day:.2f}, StdDev: ${std_fixed_comp_1_day:.2f}.")
    print(f"1-Day Trips: Estimated Fixed Component (Base + 1 Day Per Diem) - Mean: ${avg_fixed_comp_1_day:.2f}, Median: ${median_fixed_comp_1_day:.2f}, StdDev: ${std_fixed_comp_1_day:.2f}.")
    
    # The linear model's (intercept + 1*per_diem_coeff) = 266.71 + 50.05 = $316.76.
    # The median_fixed_comp_1_day is likely much higher.
    if median_fixed_comp_1_day > 350: # Arbitrary threshold to indicate it's significantly different
        findings.append(f"CONFIRMED: 1-Day trips likely have a distinct, larger base fixed amount (around ${median_fixed_comp_1_day:.2f}) than predicted by extending the multi-day linear formula.")
        print(f"This suggests a specific base amount for 1-day trips around ${median_fixed_comp_1_day:.2f}.")
        # Store this for later use in refined formula
        df.loc[df['trip_duration_days'] == 1, 'refined_base_component'] = median_fixed_comp_1_day
    else:
        # Use linear model's base for 1-day trips if no strong signal for a different one
        df.loc[df['trip_duration_days'] == 1, 'refined_base_component'] = 266.71 + 50.05 * 1


def refine_mileage_tiers(df, findings):
    print("\n--- 4. Refining Mileage Tiers ---")
    # Mileage_Reimbursement_Estimate = expected_output - refined_base_component - current_best_receipt_contrib
    # refined_base_component for days > 1 is from linear: 266.71 + 50.05 * days
    # refined_base_component for day == 1 is from analyze_one_day_trips_deep_dive
    
    df.loc[df['trip_duration_days'] > 1, 'refined_base_component'] = 266.71 + (50.05 * df.loc[df['trip_duration_days'] > 1, 'trip_duration_days'])
    # Ensure 'refined_base_component' is filled for 1-day trips if analyze_one_day_trips_deep_dive wasn't called or didn't set it
    if 'refined_base_component' not in df.columns or df['refined_base_component'].isnull().any():
         df.loc[df['trip_duration_days'] == 1, 'refined_base_component'] = 266.71 + 50.05 * 1 # Fallback

    df['mileage_reimbursement_estimate_refined'] = df['expected_output'] - df['refined_base_component'] - df['current_best_receipt_contrib']
    df.loc[df['miles_traveled'] > 0, 'estimated_rpm_refined'] = df['mileage_reimbursement_estimate_refined'] / df['miles_traveled']
    df.loc[df['miles_traveled'] == 0, 'estimated_rpm_refined'] = 0 # Avoid NaN/inf if miles is 0

    # Tier 1: 0 < miles <= 100
    rpm_tier1 = df[(df['miles_traveled'] > 0) & (df['miles_traveled'] <= 100)]['estimated_rpm_refined'].median()
    # Tier 2: 100 < miles <= 500
    rpm_tier2 = df[(df['miles_traveled'] > 100) & (df['miles_traveled'] <= 500)]['estimated_rpm_refined'].median()
    # Tier 3: miles > 500
    rpm_tier3 = df[df['miles_traveled'] > 500]['estimated_rpm_refined'].median()

    findings.append(f"Refined Mileage Tier RPMs (Median): Tier 1 (0-100mi): ${rpm_tier1:.3f}, Tier 2 (101-500mi): ${rpm_tier2:.3f}, Tier 3 (>500mi): ${rpm_tier3:.3f}")
    print(f"Refined Mileage Tier RPMs (Median): Tier 1 (0-100mi): ${rpm_tier1:.3f}, Tier 2 (101-500mi): ${rpm_tier2:.3f}, Tier 3 (>500mi): ${rpm_tier3:.3f}")

    # The $40/mile for short trips was an artifact. The actual RPM after accounting for other components might be low or even negative if base is high.
    # If rpm_tier1 is very high/unstable, it might mean a fixed bonus for any travel (miles > 0) instead of per-mile for this tier.
    # Or, the 1-day trip's large fixed component already handles short travel for 1-day trips.
    # For multi-day short mileage trips:
    multi_day_short_mileage = df[(df['trip_duration_days'] > 1) & (df['miles_traveled'] > 0) & (df['miles_traveled'] <= 100)]
    if not multi_day_short_mileage.empty:
        median_mileage_contrib_md_short = multi_day_short_mileage['mileage_reimbursement_estimate_refined'].median()
        findings.append(f"Multi-day, Short Mileage (0-100mi): Median estimated mileage contribution is ${median_mileage_contrib_md_short:.2f}. This could be a fixed bonus for these trips.")
        print(f"Multi-day, Short Mileage (0-100mi): Median estimated mileage contribution is ${median_mileage_contrib_md_short:.2f}.")
        # Store these refined rates for final formula
        df['final_mileage_rate_tier1'] = rpm_tier1 if not pd.isna(rpm_tier1) else 0.0 # Fallback
        df['final_mileage_bonus_md_short'] = median_mileage_contrib_md_short if not pd.isna(median_mileage_contrib_md_short) else 0.0
    else:
        df['final_mileage_rate_tier1'] = 0.0
        df['final_mileage_bonus_md_short'] = 0.0


    df['final_mileage_rate_tier2'] = rpm_tier2 if not pd.isna(rpm_tier2) else 0.45 # Fallback to overall linear
    df['final_mileage_rate_tier3'] = rpm_tier3 if not pd.isna(rpm_tier3) else 0.40 # Fallback slightly lower

def synthesize_final_rules(df, findings_list):
    print("\n--- 5. Synthesizing Final Rules & Formula Parameters ---")
    
    # Rule for 1-Day Trips Base
    base_1_day = df.loc[df['trip_duration_days'] == 1, 'refined_base_component'].median() # Should be from analyze_one_day_trips
    if pd.isna(base_1_day): base_1_day = 316.76 # Fallback to linear
    findings_list.append(f"FINAL RULE: For 1-Day Trips, Base Fixed Component = ${base_1_day:.2f}")

    # Rule for Multi-Day Trips Base
    # Using linear model: intercept (266.71) + per_diem_coeff (50.05) * days
    findings_list.append(f"FINAL RULE: For Multi-Day Trips, Base Component = $266.71 + ($50.05 * trip_duration_days)")

    # Rule for Receipts
    receipt_cap_strategy = next((s for s in findings_list if "Selected Receipt Strategy" in s), "None")
    receipt_contrib_cap = None
    rapd_cap = None
    receipt_multiplier = 0.38 # Default from linear

    if "Total Contribution Cap" in receipt_cap_strategy:
        try:
            receipt_contrib_cap = float(receipt_cap_strategy.split("Value: ")[1].split(")")[0])
            findings_list.append(f"FINAL RULE: Receipts. Multiplier = {receipt_multiplier}. Contribution from (Multiplier * total_receipts_amount) is CAPPED at ${receipt_contrib_cap:.2f}.")
        except: findings_list.append("FINAL RULE: Receipts. Multiplier = 0.38. Error parsing contribution cap.")
    elif "RAPD Cap" in receipt_cap_strategy:
        try:
            rapd_cap = float(receipt_cap_strategy.split("Value: ")[1].split(")")[0])
            findings_list.append(f"FINAL RULE: Receipts. Multiplier = {receipt_multiplier}. Effective total_receipts_amount is CAPPED at (${rapd_cap:.2f} * trip_duration_days).")
        except: findings_list.append("FINAL RULE: Receipts. Multiplier = 0.38. Error parsing RAPD cap.")
    else:
        findings_list.append(f"FINAL RULE: Receipts. Multiplier = {receipt_multiplier} * total_receipts_amount. No cap applied based on current analysis improvement.")

    # Rule for Mileage
    rate_tier1 = df['final_mileage_rate_tier1'].iloc[0] if 'final_mileage_rate_tier1' in df and not df.empty else 0.0
    bonus_md_short = df['final_mileage_bonus_md_short'].iloc[0] if 'final_mileage_bonus_md_short' in df and not df.empty else 0.0
    rate_tier2 = df['final_mileage_rate_tier2'].iloc[0] if 'final_mileage_rate_tier2' in df and not df.empty else 0.45
    rate_tier3 = df['final_mileage_rate_tier3'].iloc[0] if 'final_mileage_rate_tier3' in df and not df.empty else 0.40

    findings_list.append(f"FINAL RULE: Mileage Tiers:")
    findings_list.append(f"  - Tier 1 (0 < miles <= 100):")
    findings_list.append(f"    - If 1-day trip: Rate likely ${rate_tier1:.3f}/mile (or this is part of the large 1-day base).")
    findings_list.append(f"    - If multi-day trip: Fixed bonus of approx ${bonus_md_short:.2f} PLUS potentially a small per-mile rate like ${rate_tier1:.3f}/mile for actual miles.")
    findings_list.append(f"  - Tier 2 (100 < miles <= 500): Rate approx ${rate_tier2:.3f}/mile.")
    findings_list.append(f"  - Tier 3 (miles > 500): Rate approx ${rate_tier3:.3f}/mile.")
    findings_list.append(f"  - If miles == 0: Mileage contribution is $0.")
    
    # Construct a test formula based on these rules
    def apply_final_formula(row):
        # Base
        if row['trip_duration_days'] == 1:
            current_base = base_1_day
        else:
            current_base = 266.71 + (50.05 * row['trip_duration_days'])

        # Receipts
        effective_receipts = row['total_receipts_amount']
        if rapd_cap: # RAPD Cap strategy
            effective_receipts = min(row['total_receipts_amount'], rapd_cap * row['trip_duration_days'])
        
        receipt_contribution = receipt_multiplier * effective_receipts
        if receipt_contrib_cap: # Total Contribution Cap strategy
            receipt_contribution = min(receipt_contribution, receipt_contrib_cap)

        # Mileage
        mileage_contribution = 0
        if row['miles_traveled'] == 0:
            mileage_contribution = 0
        elif row['miles_traveled'] <= 100:
            if row['trip_duration_days'] == 1:
                 # For 1-day short trips, the large base_1_day might already cover it, or a small rate applies.
                 # This part is the most uncertain. Let's use rate_tier1.
                mileage_contribution = row['miles_traveled'] * rate_tier1 
            else: # Multi-day short mileage
                mileage_contribution = bonus_md_short + (row['miles_traveled'] * rate_tier1) # Bonus + per-mile
        elif row['miles_traveled'] <= 500:
            mileage_contribution = row['miles_traveled'] * rate_tier2
        else: # miles > 500
            mileage_contribution = row['miles_traveled'] * rate_tier3
            
        return current_base + receipt_contribution + mileage_contribution

    df['predicted_final_formula'] = df.apply(apply_final_formula, axis=1)
    df['residual_final_formula'] = df['expected_output'] - df['predicted_final_formula']
    final_avg_abs_error = df['residual_final_formula'].abs().mean()
    
    initial_avg_abs_error = df['residual_linear'].abs().mean()
    findings_list.append(f"Applying Synthesized Rules: Initial Avg Abs Error (Linear): ${initial_avg_abs_error:.2f}. New Avg Abs Error: ${final_avg_abs_error:.2f}.")
    print(f"Applying Synthesized Rules: Initial Avg Abs Error (Linear): ${initial_avg_abs_error:.2f}. New Avg Abs Error: ${final_avg_abs_error:.2f}.")


def main():
    findings_summary = []
    df_loaded = load_data()
    df_processed = create_derived_features(df_loaded)

    analyze_high_error_cases(df_processed, findings_summary)
    investigate_receipt_caps_penalties(df_processed, findings_summary) # df_processed gets 'current_best_predicted' etc.
    analyze_one_day_trips_deep_dive(df_processed, findings_summary) # df_processed gets 'refined_base_component' for day 1
    refine_mileage_tiers(df_processed, findings_summary) # df_processed gets 'final_mileage_rate_tierX'

    synthesize_final_rules(df_processed, findings_summary)

    print("\n\n--- Summary of Advanced Pattern Discovery Findings & Derived Rules ---")
    for i, finding in enumerate(findings_summary):
        print(f"{i+1}. {finding}")
    
    # Save the dataframe with all calculations for inspection if needed
    # df_processed.to_csv("advanced_analysis_output.csv", index=False)
    # print("\nSaved detailed analysis to advanced_analysis_output.csv")

    print("\n--- End of Advanced Pattern Discovery Script ---")

if __name__ == '__main__':
    main()

