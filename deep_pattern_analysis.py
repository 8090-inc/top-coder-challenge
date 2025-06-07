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


# --- Helper Functions (from analyze_data.py or simplified) ---
def load_data(file_path="public_cases.json"):
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

def create_derived_features(df):
    df_c = df.copy()
    df_c['miles_per_day'] = df_c['miles_traveled'] / df_c['trip_duration_days'].replace(0, np.nan)
    df_c['receipt_amount_per_day'] = df_c['total_receipts_amount'] / df_c['trip_duration_days'].replace(0, np.nan)
    df_c['reimbursement_per_day'] = df_c['expected_output'] / df_c['trip_duration_days'].replace(0, np.nan)
    df_c['reimbursement_per_mile'] = np.where(df_c['miles_traveled'] > 0, df_c['expected_output'] / df_c['miles_traveled'], np.nan)
    df_c['reimbursement_minus_receipts'] = df_c['expected_output'] - df_c['total_receipts_amount']
    df_c['receipt_cents'] = ((df_c['total_receipts_amount'] * 100) % 100).round().astype(int)
    return df_c

# --- Deep Dive Analysis Functions ---

def analyze_short_mileage_anomaly(df, findings):
    print("\n--- Deep Dive: Short Mileage Anomaly (<= 100 miles) ---")
    short_mileage_trips = df[df['miles_traveled'] <= 100].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    if short_mileage_trips.empty:
        findings.append("Short Mileage: No trips with <= 100 miles found.")
        print("No trips with <= 100 miles found.")
        return

    # Initial finding: Avg reimbursement/mile for <= 100 miles: $40.392
    # This is ((expected_output) / miles_traveled).mean() for these trips
    # Let's re-calculate this carefully, excluding miles_traveled == 0
    short_mileage_trips_gt_0_miles = short_mileage_trips[short_mileage_trips['miles_traveled'] > 0]
    if not short_mileage_trips_gt_0_miles.empty:
        avg_rpm_short = short_mileage_trips_gt_0_miles['reimbursement_per_mile'].mean()
        findings.append(f"Short Mileage: Avg Reimbursement/Mile for (0 < miles <= 100): ${avg_rpm_short:.3f}.")
        print(f"Avg Reimbursement/Mile for (0 < miles <= 100): ${avg_rpm_short:.3f}")

        # Hypothesis: expected_output = base_fixed_for_travel + per_diem_component + receipt_component + small_mileage_rate * miles
        # Or, expected_output = per_diem_component + receipt_component + large_fixed_mileage_bonus_for_short_trips
        
        # Let's try to isolate the "mileage component" for these short trips
        # MileageComponent = ExpectedOutput - (PerDiemGuess * Days) - (ReceiptsGuess * TotalReceipts)
        # Using $110/day as per_diem_guess and 0.7 * receipts as receipt_guess (from correlations)
        per_diem_guess = 110 
        receipt_multiplier_guess = 0.7 # From initial correlation
        
        short_mileage_trips_gt_0_miles.loc[:, 'estimated_per_diem_cost'] = short_mileage_trips_gt_0_miles['trip_duration_days'] * per_diem_guess
        short_mileage_trips_gt_0_miles.loc[:, 'estimated_receipt_reimbursement'] = short_mileage_trips_gt_0_miles['total_receipts_amount'] * receipt_multiplier_guess
        short_mileage_trips_gt_0_miles.loc[:, 'residual_for_mileage'] = short_mileage_trips_gt_0_miles['expected_output'] - short_mileage_trips_gt_0_miles['estimated_per_diem_cost'] - short_mileage_trips_gt_0_miles['estimated_receipt_reimbursement']
        short_mileage_trips_gt_0_miles.loc[:, 'residual_rpm'] = short_mileage_trips_gt_0_miles['residual_for_mileage'] / short_mileage_trips_gt_0_miles['miles_traveled']

        avg_residual_rpm = short_mileage_trips_gt_0_miles['residual_rpm'].median() # Median robust to outliers
        findings.append(f"Short Mileage: Median 'Residual RPM' (after $110/day and 0.7*receipts): ${avg_residual_rpm:.3f}.")
        print(f"Median 'Residual RPM' (after $110/day and 0.7*receipts): ${avg_residual_rpm:.3f}")
        print("This 'residual RPM' could be the actual per-mile rate for short trips, or part of a fixed bonus.")

        # Check if it's a large fixed bonus + small per mile rate
        # If residual_for_mileage is fairly constant, it's a fixed bonus.
        median_residual_for_mileage = short_mileage_trips_gt_0_miles['residual_for_mileage'].median()
        findings.append(f"Short Mileage: Median 'Residual for Mileage' (fixed bonus guess): ${median_residual_for_mileage:.2f}.")
        print(f"Median 'Residual for Mileage' (potential fixed bonus for short travel): ${median_residual_for_mileage:.2f}")
        print("If this 'Residual for Mileage' is fairly constant across short mileage trips, it could be a fixed bonus for any travel <=100 miles.")
        print("Rule Suggestion: For miles_traveled <= 100 AND miles_traveled > 0, add a bonus of approx. ${median_residual_for_mileage:.2f}, then add a smaller per-mile rate for the actual miles.")
    else:
        findings.append("Short Mileage: No trips with 0 < miles <= 100 found to calculate RPM.")
        print("No trips with 0 < miles <= 100 found to calculate RPM.")


def analyze_inverted_spending_hypothesis(df, findings):
    print("\n--- Deep Dive: Kevin's Inverted Spending Hypothesis ---")
    # Initial finding: Higher spending per day led to higher RPD, contrary to Kevin.
    
    df_c = df.copy()
    df_c['trip_length_category'] = pd.cut(df_c['trip_duration_days'], 
                                          bins=[0, 3, 6, 14], 
                                          labels=['Short (1-3d)', 'Medium (4-6d)', 'Long (7-14d)'], 
                                          right=True)

    for category in ['Short (1-3d)', 'Medium (4-6d)', 'Long (7-14d)']:
        category_df = df_c[df_c['trip_length_category'] == category].copy()
        if category_df.empty:
            print(f"No data for trip category: {category}")
            continue

        # Create bins for receipt_amount_per_day
        # Ensure bins are sensible for the data range in each category
        max_rapd = category_df['receipt_amount_per_day'].max()
        if pd.isna(max_rapd) or max_rapd == 0:
            print(f"Not enough receipt_amount_per_day variation in {category} to analyze.")
            continue
            
        # Define bins dynamically or use fixed ones if appropriate
        # Example: 0-50, 50-100, 100-150, 150+
        bins = [0, 50, 100, 150, np.inf if max_rapd > 150 else max_rapd + 1]
        labels = ['<$50/day', '$50-100/day', '$100-150/day', '>$150/day']
        
        # Adjust bins if max_rapd is low
        if max_rapd < 150:
            bins = [0, max_rapd/3, 2*max_rapd/3, max_rapd +1]
            labels = ['Low RAPD', 'Mid RAPD', 'High RAPD']
        if max_rapd < 50: # very low, maybe just two bins
             bins = [0, max_rapd/2, max_rapd+1]
             labels = ['Lower RAPD', 'Higher RAPD']


        category_df.loc[:, 'rapd_bin'] = pd.cut(category_df['receipt_amount_per_day'],
                                            bins=bins,
                                            labels=labels[:len(bins)-1], # Ensure labels match number of bins
                                            right=False, include_lowest=True)
        
        avg_rpd_by_rapd_bin = category_df.groupby('rapd_bin')['reimbursement_per_day'].mean()
        
        print(f"\nCategory: {category}")
        print("Average Reimbursement per Day (RPD) by Receipt Amount per Day (RAPD) bin:")
        print(avg_rpd_by_rapd_bin)
        
        if not avg_rpd_by_rapd_bin.empty:
            # Check if RPD generally increases with RAPD bin
            is_increasing = all(avg_rpd_by_rapd_bin.iloc[i] <= avg_rpd_by_rapd_bin.iloc[i+1] for i in range(len(avg_rpd_by_rapd_bin)-1))
            if is_increasing:
                findings.append(f"Spending Hypothesis ({category}): Confirmed - Higher RAPD leads to higher RPD.")
                print(f"Finding ({category}): Higher RAPD generally leads to higher RPD, contradicting Kevin's optimization for lower spending.")
            else:
                findings.append(f"Spending Hypothesis ({category}): Mixed - Relationship between RAPD and RPD is not strictly increasing. Details: {avg_rpd_by_rapd_bin.to_dict()}")
                print(f"Finding ({category}): Relationship between RAPD and RPD is not strictly increasing. Review bins.")
    
    # Overall check:
    df_c['rapd_overall_bin'] = pd.qcut(df_c['receipt_amount_per_day'], q=4, duplicates='drop', labels=['Q1 RAPD', 'Q2 RAPD', 'Q3 RAPD', 'Q4 RAPD'] if df_c['receipt_amount_per_day'].nunique() >=4 else False)
    if 'rapd_overall_bin' in df_c.columns and df_c['rapd_overall_bin'].notna().any():
        overall_avg_rpd_by_rapd_bin = df_c.groupby('rapd_overall_bin')['reimbursement_per_day'].mean()
        print("\nOverall Average RPD by RAPD Quartile:")
        print(overall_avg_rpd_by_rapd_bin)
        if not overall_avg_rpd_by_rapd_bin.empty and all(overall_avg_rpd_by_rapd_bin.iloc[i] <= overall_avg_rpd_by_rapd_bin.iloc[i+1] for i in range(len(overall_avg_rpd_by_rapd_bin)-1)):
             findings.append(f"Spending Hypothesis (Overall): Confirmed - Higher RAPD (quartiles) leads to higher RPD.")


def attempt_formula_discovery(df, findings):
    print("\n--- Deep Dive: Formula Discovery (Linear Regression Attempt) ---")
    df_reg = df[['total_receipts_amount', 'trip_duration_days', 'miles_traveled', 'expected_output']].copy()
    df_reg = df_reg.dropna() # Ensure no NaNs for regression

    if df_reg.shape[0] < 4: # Need at least as many data points as features + intercept
        findings.append("Formula Discovery: Not enough data points for linear regression.")
        print("Not enough data points for linear regression.")
        return

    # Prepare data for np.linalg.lstsq
    X = df_reg[['total_receipts_amount', 'trip_duration_days', 'miles_traveled']].values
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add intercept column
    y = df_reg['expected_output'].values

    try:
        coeffs, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
        intercept, coeff_receipts, coeff_days, coeff_miles = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

        formula_str = (f"Base Formula Suggestion: expected_output = "
                       f"{intercept:.2f} (intercept) + "
                       f"{coeff_receipts:.2f} * total_receipts_amount + "
                       f"{coeff_days:.2f} * trip_duration_days + "
                       f"{coeff_miles:.2f} * miles_traveled")
        findings.append(formula_str)
        print(formula_str)

        # Check per diem from this formula:
        # The per diem is coeff_days. The intercept is a fixed base amount for any trip.
        findings.append(f"Formula Implied Per Diem (coeff_days): ${coeff_days:.2f}/day.")
        print(f"Formula Implied Per Diem (coeff_days): ${coeff_days:.2f}/day.")
        findings.append(f"Formula Implied Base Fixed Amount (intercept): ${intercept:.2f}.")
        print(f"Formula Implied Base Fixed Amount (intercept): ${intercept:.2f}.")
        findings.append(f"Formula Implied Receipt Multiplier: {coeff_receipts:.2f} (matches 0.70 correlation direction).")
        print(f"Formula Implied Receipt Multiplier: {coeff_receipts:.2f}.")
        findings.append(f"Formula Implied Mileage Rate: ${coeff_miles:.2f}/mile (this is an overall average, likely tiered).")
        print(f"Formula Implied Mileage Rate: ${coeff_miles:.2f}/mile (this is an overall average, likely tiered).")

        # Analyze residuals
        df_reg.loc[:,'predicted_output_linear'] = X @ coeffs
        df_reg.loc[:,'residual_linear'] = df_reg['expected_output'] - df_reg['predicted_output_linear']
        
        plt.figure(figsize=(8,5))
        sns.histplot(df_reg['residual_linear'], kde=True)
        plt.title('Distribution of Residuals from Base Linear Model')
        plt.xlabel('Residual (Expected - Predicted)')
        plt.savefig('linear_model_residuals.png')
        plt.close()
        print("Saved linear_model_residuals.png")
        findings.append("Linear Model Residuals: Check 'linear_model_residuals.png'. Large spread or patterns indicate need for non-linearities/tiers/bonuses.")

    except np.linalg.LinAlgError as e:
        findings.append(f"Formula Discovery: Linear regression failed. Error: {e}")
        print(f"Linear regression failed. Error: {e}")


def analyze_one_day_trips(df, findings):
    print("\n--- Deep Dive: 1-Day Trip Anomaly ---")
    one_day_trips = df[df['trip_duration_days'] == 1].copy()

    if one_day_trips.empty:
        findings.append("1-Day Trips: No 1-day trips found.")
        print("No 1-day trips found.")
        return

    avg_output_1_day = one_day_trips['expected_output'].mean()
    # Initial analysis showed RPD for 1-day trips: $873.55. This is the same as total for 1-day.
    findings.append(f"1-Day Trips: Average total reimbursement: ${avg_output_1_day:.2f}.")
    print(f"Average total reimbursement for 1-day trips: ${avg_output_1_day:.2f}.")

    # From linear model: expected_output = intercept + coeff_receipts*R + coeff_days*1 + coeff_miles*M
    # If intercept is e.g. $400 and coeff_days is $100, then 1-day trips get $500 base + R + M components.
    # The $873.55 is an average. Let's see the components for these trips using the derived coeffs.
    # Using coeffs from attempt_formula_discovery (if available, otherwise use placeholders)
    # This part is tricky as coeffs are global. For now, just describe the structure.
    
    print("For 1-day trips, the 'intercept' from a general linear model would be a large part of their reimbursement.")
    print("The high RPD for 1-day trips ($873.55) compared to, say, 2-day trips ($523.12 RPD) suggests a large fixed component that isn't scaled by days, or scales non-linearly for the first day.")
    
    # Let's check the structure of these 1-day trips
    print("\nStats for 1-day trips:")
    print(one_day_trips[['miles_traveled', 'total_receipts_amount', 'expected_output']].describe())

    # Are these 1-day trips also the ones with <=100 miles that get the $40/mile?
    one_day_short_mileage = one_day_trips[one_day_trips['miles_traveled'] <= 100]
    if not one_day_short_mileage.empty:
        avg_output_1d_short_miles = one_day_short_mileage['expected_output'].mean()
        findings.append(f"1-Day Trips (<=100 miles): Avg reimbursement ${avg_output_1d_short_miles:.2f} ({len(one_day_short_mileage)} cases).")
        print(f"1-Day Trips (<=100 miles): Avg reimbursement ${avg_output_1d_short_miles:.2f} ({len(one_day_short_mileage)} cases).")
    
    one_day_long_mileage = one_day_trips[one_day_trips['miles_traveled'] > 100]
    if not one_day_long_mileage.empty:
        avg_output_1d_long_miles = one_day_long_mileage['expected_output'].mean()
        findings.append(f"1-Day Trips (>100 miles): Avg reimbursement ${avg_output_1d_long_miles:.2f} ({len(one_day_long_mileage)} cases).")
        print(f"1-Day Trips (>100 miles): Avg reimbursement ${avg_output_1d_long_miles:.2f} ({len(one_day_long_mileage)} cases).")
    
    findings.append("1-Day Trips Rule Suggestion: Treat 1-day trips as a special case. Their calculation might be: Fixed_Base_1Day + Receipt_Component_1Day + Mileage_Component_1Day. The Mileage_Component might further split based on <=100 miles vs >100 miles.")


def analyze_mileage_tiers_deeper(df, findings):
    print("\n--- Deep Dive: Mileage Tiers ---")
    df_miles_positive = df[df['miles_traveled'] > 0].copy()
    if df_miles_positive.empty:
        findings.append("Mileage Tiers: No trips with miles > 0 found.")
        print("No trips with miles > 0 found.")
        return

    # 1. Short trips (0 < miles <= 100)
    # Already covered by analyze_short_mileage_anomaly. The key is the $40.392/mile or a fixed bonus.
    # Let's assume the "Residual for Mileage" of ~$300-$400 found earlier is a fixed bonus for any travel.
    # And then a smaller per-mile rate.
    # Example: Fixed Travel Bonus = $350 (if miles > 0 and miles <=100)
    #          Mileage Rate for these short trips = (Residual for Mileage - Fixed Travel Bonus) / miles_traveled
    # This needs more iterative refinement.

    # 2. Medium Mileage (e.g., 100 < miles <= 500)
    medium_mileage_df = df_miles_positive[(df_miles_positive['miles_traveled'] > 100) & (df_miles_positive['miles_traveled'] <= 500)].copy()
    if not medium_mileage_df.empty:
        # Estimate mileage rate after accounting for per diem and receipts
        per_diem_guess = 110
        receipt_multiplier_guess = 0.7 # From correlation
        medium_mileage_df.loc[:, 'non_mileage_cost_estimate'] = (medium_mileage_df['trip_duration_days'] * per_diem_guess) + \
                                                              (medium_mileage_df['total_receipts_amount'] * receipt_multiplier_guess)
        medium_mileage_df.loc[:, 'mileage_reimbursement_estimate'] = medium_mileage_df['expected_output'] - medium_mileage_df['non_mileage_cost_estimate']
        medium_mileage_df.loc[:, 'estimated_rpm'] = medium_mileage_df['mileage_reimbursement_estimate'] / medium_mileage_df['miles_traveled']
        
        avg_rpm_medium = medium_mileage_df[medium_mileage_df['estimated_rpm'] > 0]['estimated_rpm'].median() # Median, positive
        findings.append(f"Mileage Tiers (100 < miles <= 500): Estimated median RPM (after $110/day & 0.7*receipts): ${avg_rpm_medium:.3f}")
        print(f"For 100 < miles <= 500: Estimated median RPM (after $110/day & 0.7*receipts): ${avg_rpm_medium:.3f}")
    else:
        print("No trips found for 100 < miles <= 500.")

    # 3. Long Mileage (miles > 500)
    long_mileage_df = df_miles_positive[df_miles_positive['miles_traveled'] > 500].copy()
    if not long_mileage_df.empty:
        long_mileage_df.loc[:, 'non_mileage_cost_estimate'] = (long_mileage_df['trip_duration_days'] * per_diem_guess) + \
                                                            (long_mileage_df['total_receipts_amount'] * receipt_multiplier_guess)
        long_mileage_df.loc[:, 'mileage_reimbursement_estimate'] = long_mileage_df['expected_output'] - long_mileage_df['non_mileage_cost_estimate']
        long_mileage_df.loc[:, 'estimated_rpm'] = long_mileage_df['mileage_reimbursement_estimate'] / long_mileage_df['miles_traveled']

        avg_rpm_long = long_mileage_df[long_mileage_df['estimated_rpm'] > 0]['estimated_rpm'].median()
        findings.append(f"Mileage Tiers (miles > 500): Estimated median RPM (after $110/day & 0.7*receipts): ${avg_rpm_long:.3f}")
        print(f"For miles > 500: Estimated median RPM (after $110/day & 0.7*receipts): ${avg_rpm_long:.3f}")
        
        if not medium_mileage_df.empty and avg_rpm_long < avg_rpm_medium:
            print("Observation: RPM for >500 miles seems lower than for 100-500 miles, suggesting diminishing returns or tiered rates.")
            findings.append("Mileage Tiers: RPM for >500 miles seems lower than for 100-500 miles.")
    else:
        print("No trips found for miles > 500.")
    
    findings.append("Mileage Rule Suggestion: Tiered system. Tier 1 (0 < miles <= 100): Potential fixed bonus + low per-mile rate. Tier 2 (100 < miles <= X): Higher per-mile rate (e.g., $0.50-$0.60). Tier 3 (miles > X): Lower per-mile rate. X needs to be found, maybe around 500 miles.")


def main():
    findings = []
    df_raw = load_data()
    df = create_derived_features(df_raw)

    print("--- Deep Pattern Analysis Script ---")

    # 1. SHOCKING: Avg reimbursement/mile for <= 100 miles: $40.392
    analyze_short_mileage_anomaly(df, findings)

    # 2. INVERTED HYPOTHESIS: Kevin's spending recommendations are backwards
    analyze_inverted_spending_hypothesis(df, findings)

    # 3. STRONG CORRELATION & Formula Discovery
    # 4. COMPONENT ISOLATION: Estimated per diem ~$110/day
    attempt_formula_discovery(df, findings) # This covers per diem and correlations implicitly

    # Trip duration analysis (1-day trips get $873/day?!)
    analyze_one_day_trips(df, findings)
    
    # Mileage tier analysis (the $40/mile for short trips suggests a different calculation)
    analyze_mileage_tiers_deeper(df, findings)

    # Receipt processing rules (0.70 correlation is very strong)
    # The linear formula attempt gives a coefficient for receipts.
    # Further analysis could look for non-linearities in receipt impact.
    # For now, the linear coefficient is a starting point.
    print("\n--- Receipt Processing (Recap from Linear Model) ---")
    # Find the receipt coefficient from findings if attempt_formula_discovery ran
    receipt_coeff_str = next((s for s in findings if "Formula Implied Receipt Multiplier" in s), None)
    if receipt_coeff_str:
        print(receipt_coeff_str)
        findings.append("Receipt Rule Suggestion: A significant portion of reimbursement is directly tied to total_receipts_amount, likely a multiplier around the one found in the linear model. Check for caps or non-linear effects if linear model residuals are large for high receipt values.")
    else:
        findings.append("Receipt Rule Suggestion: total_receipts_amount is highly correlated (0.70). A simple multiplier is a good first guess. Refine if linear model is poor.")


    print("\n\n--- Summary of Deep Dive Actionable Findings & Formulas ---")
    if not findings:
        print("No specific actionable findings generated from deep dive.")
    for i, finding in enumerate(findings):
        print(f"{i+1}. {finding}")
    
    print("\n--- End of Deep Pattern Analysis ---")

if __name__ == '__main__':
    main()

