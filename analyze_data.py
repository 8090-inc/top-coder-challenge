import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

def load_data(file_path="public_cases.json"):
    """Loads data from the JSON file into a pandas DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    inputs = [item['input'] for item in data]
    outputs = [item['expected_output'] for item in data]
    
    df = pd.DataFrame(inputs)
    df['expected_output'] = outputs
    
    # Ensure correct data types, especially for potentially integer inputs
    df['trip_duration_days'] = df['trip_duration_days'].astype(int)
    # Miles traveled can be float in public_cases.json, despite README saying int. Analyze as is.
    df['miles_traveled'] = df['miles_traveled'].astype(float) 
    df['total_receipts_amount'] = df['total_receipts_amount'].astype(float)
    df['expected_output'] = df['expected_output'].astype(float)
    
    return df

def create_derived_features(df):
    """Creates derived features based on interviews and common sense."""
    df_c = df.copy()
    
    # Basic ratios
    # Handle division by zero by replacing 0 with np.nan in denominator, then result will be np.nan
    df_c['miles_per_day'] = df_c['miles_traveled'] / df_c['trip_duration_days']
    df_c['receipt_amount_per_day'] = df_c['total_receipts_amount'] / df_c['trip_duration_days']
    df_c['reimbursement_per_day'] = df_c['expected_output'] / df_c['trip_duration_days']
    
    df_c['reimbursement_per_mile'] = np.where(df_c['miles_traveled'] > 0, df_c['expected_output'] / df_c['miles_traveled'], np.nan)
    
    # Components (very rough estimates for now)
    df_c['reimbursement_minus_receipts'] = df_c['expected_output'] - df_c['total_receipts_amount']
    df_c['reimbursement_per_day_minus_receipts_per_day'] = df_c['reimbursement_per_day'] - df_c['receipt_amount_per_day']

    # Cents from receipts (for rounding bug hypothesis)
    df_c['receipt_cents'] = ((df_c['total_receipts_amount'] * 100) % 100).round().astype(int)

    return df_c

def plot_distributions(df, features, title_prefix=""):
    """Plots distributions for a list of features."""
    num_features = len(features)
    plt.figure(figsize=(15, num_features * 3))
    for i, feature in enumerate(features):
        plt.subplot(num_features, 2, 2*i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'{title_prefix}Distribution of {feature}')
        
        plt.subplot(num_features, 2, 2*i + 2)
        sns.boxplot(x=df[feature])
        plt.title(f'{title_prefix}Boxplot of {feature}')
    plt.tight_layout()
    plt.show()

def analyze_trip_duration(df, findings):
    """Analyzes impact of trip duration."""
    print("\n--- Trip Duration Analysis ---")
    
    # 5-day trip bonus (Lisa, Marcus)
    avg_reimbursement_per_day = df.groupby('trip_duration_days')['reimbursement_per_day'].mean()
    avg_reimbursement = df.groupby('trip_duration_days')['expected_output'].mean()
    
    print("Average reimbursement per day by trip duration:\n", avg_reimbursement_per_day)
    print("Average total reimbursement by trip duration:\n", avg_reimbursement)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='trip_duration_days', y='reimbursement_per_day', data=df)
    plt.title('Reimbursement per Day vs. Trip Duration')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='trip_duration_days', y='expected_output', data=df)
    plt.title('Total Reimbursement vs. Trip Duration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if 5 in avg_reimbursement_per_day.index:
        rpd_5_day = avg_reimbursement_per_day.loc[5]
        other_days_avg_rpd = avg_reimbursement_per_day[avg_reimbursement_per_day.index != 5].mean()
        findings.append(f"5-day trips: Avg RPD ${rpd_5_day:.2f}. Other days avg RPD: ${other_days_avg_rpd:.2f}.")
        if rpd_5_day > other_days_avg_rpd:
            findings.append("Suggests a bonus for 5-day trips.")
    
    # Sweet spot around 4-6 days (Jennifer) or 5-6 days (Marcus)
    # Marcus: 8-day trip was incredible.
    # This is somewhat covered by the boxplots. Look for non-linearities.
    # The plots will visually show this. For example, if days 4,5,6 are higher than 1,2,3 and 7,8,9...

def analyze_mileage(df, findings):
    """Analyzes impact of miles traveled."""
    print("\n--- Mileage Analysis ---")
    
    # Mileage tiers/thresholds (Lisa, Marcus, Kevin)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    sns.scatterplot(x='miles_traveled', y='expected_output', data=df, alpha=0.5)
    plt.title('Total Reimbursement vs. Miles Traveled')
    
    plt.subplot(1, 3, 2)
    # Filter out cases with 0 miles for reimbursement_per_mile analysis if they cause issues
    df_miles_positive = df[df['miles_traveled'] > 0]
    sns.scatterplot(x='miles_traveled', y='reimbursement_per_mile', data=df_miles_positive, alpha=0.5)
    plt.title('Reimbursement per Mile vs. Miles Traveled')
    plt.ylim(0, 2) # Cap y-axis for better visualization if outliers exist

    # Lisa: "First 100 miles or so, you get the full rate—like 58 cents per mile."
    rpm_under_100 = df_miles_positive[df_miles_positive['miles_traveled'] <= 100]['reimbursement_per_mile'].mean()
    rpm_over_100 = df_miles_positive[df_miles_positive['miles_traveled'] > 100]['reimbursement_per_mile'].mean()
    print(f"Avg reimbursement/mile for <= 100 miles: ${rpm_under_100:.3f}")
    findings.append(f"Avg reimbursement/mile for <= 100 miles: ${rpm_under_100:.3f}. Lisa suggested $0.58.")
    print(f"Avg reimbursement/mile for > 100 miles: ${rpm_over_100:.3f}")
    findings.append(f"Avg reimbursement/mile for > 100 miles: ${rpm_over_100:.3f}. Suggests rate drops.")

    # Marcus: 600-mile trip -> $298 (expected $350). Rate $0.496/mile.
    # Check trips around 600 miles
    trips_around_600_miles = df_miles_positive[(df_miles_positive['miles_traveled'] >= 580) & (df_miles_positive['miles_traveled'] <= 620)]
    if not trips_around_600_miles.empty:
        avg_rpm_600 = trips_around_600_miles['reimbursement_per_mile'].mean()
        print(f"Avg reimbursement/mile for trips ~600 miles: ${avg_rpm_600:.3f}")
        findings.append(f"Avg reimbursement/mile for trips ~600 miles: ${avg_rpm_600:.3f}. Marcus saw ~$0.496.")

    # Efficiency: miles_per_day (Kevin, Lisa)
    plt.subplot(1, 3, 3)
    sns.scatterplot(x='miles_per_day', y='reimbursement_per_day', data=df, alpha=0.5)
    plt.title('Reimbursement per Day vs. Miles per Day')
    plt.xlim(0, df['miles_per_day'].quantile(0.99) if not df.empty else 500) # Cap x-axis for clarity
    plt.tight_layout()
    plt.show()

    # Kevin: "Sweet spot around 180-220 miles per day"
    mpd_sweet_spot = df[(df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)]
    mpd_outside_sweet_spot = df[(df['miles_per_day'] < 180) | (df['miles_per_day'] > 220)]
    
    if not mpd_sweet_spot.empty and not mpd_outside_sweet_spot.empty:
        avg_rpd_sweet = mpd_sweet_spot['reimbursement_per_day'].mean()
        avg_rpd_outside = mpd_outside_sweet_spot['reimbursement_per_day'].mean()
        print(f"Avg RPD for miles/day 180-220: ${avg_rpd_sweet:.2f}")
        print(f"Avg RPD for miles/day outside 180-220: ${avg_rpd_outside:.2f}")
        if avg_rpd_sweet > avg_rpd_outside:
            findings.append(f"Miles/day 180-220 (Kevin's sweet spot) shows higher RPD (${avg_rpd_sweet:.2f} vs ${avg_rpd_outside:.2f}).")

def analyze_receipts(df, findings):
    """Analyzes impact of receipt amounts."""
    print("\n--- Receipt Analysis ---")
    
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    sns.scatterplot(x='total_receipts_amount', y='expected_output', data=df, alpha=0.5)
    plt.title('Total Reimbursement vs. Total Receipt Amount')
    plt.xlim(0, df['total_receipts_amount'].quantile(0.99) if not df.empty else 2500) # Cap for clarity

    # Reimbursement related to receipts (expected_output - (per_diem_guess + mileage_guess))
    # This is hard without better component estimates. For now, look at reimbursement_minus_receipts
    plt.subplot(1, 3, 2)
    sns.scatterplot(x='total_receipts_amount', y='reimbursement_minus_receipts', data=df, alpha=0.5)
    plt.title('(Reimbursement - Receipts) vs. Total Receipt Amount')
    plt.xlim(0, df['total_receipts_amount'].quantile(0.99) if not df.empty else 2500)

    plt.subplot(1, 3, 3)
    df_receipts_positive = df[df['total_receipts_amount'] > 0]
    if not df_receipts_positive.empty:
        df_receipts_positive['receipt_reimbursement_ratio'] = df_receipts_positive['expected_output'] / df_receipts_positive['total_receipts_amount']
        sns.scatterplot(x='total_receipts_amount', y='receipt_reimbursement_ratio', data=df_receipts_positive, alpha=0.5)
        plt.title('Reimbursement/Receipts Ratio vs. Total Receipt Amount')
        plt.ylim(0, 5) # Cap for clarity
        plt.xlim(0, df_receipts_positive['total_receipts_amount'].quantile(0.99) if not df_receipts_positive.empty else 2500)
    plt.tight_layout()
    plt.show()

    # Small receipt penalties (Lisa, Dave)
    # Lisa: "if you submit $50 in receipts for a multi-day trip, you're better off submitting nothing."
    # Dave: "$12 in receipts, I might get less than the per diem."
    small_receipts_multi_day = df[(df['total_receipts_amount'] > 0) & (df['total_receipts_amount'] < 50) & (df['trip_duration_days'] > 1)]
    if not small_receipts_multi_day.empty:
        avg_reimbursement_small_receipts = small_receipts_multi_day['expected_output'].mean()
        print(f"Avg reimbursement for multi-day trips with <$50 receipts: ${avg_reimbursement_small_receipts:.2f}")
        findings.append(f"Avg reimbursement for multi-day trips with <$50 receipts: ${avg_reimbursement_small_receipts:.2f}. Check if this is lower than expected base per diem + mileage.")

    # Receipt amount sweet spots/diminishing returns (Lisa)
    # Lisa: "Medium-high amounts—like $600-800—seem to get really good treatment."
    receipts_600_800 = df[(df['total_receipts_amount'] >= 600) & (df['total_receipts_amount'] <= 800)]
    if not receipts_600_800.empty and not df_receipts_positive.empty: # df_receipts_positive for 'receipt_reimbursement_ratio'
        avg_ratio_600_800 = receipts_600_800[receipts_600_800['total_receipts_amount'] > 0]['expected_output'].sum() / receipts_600_800[receipts_600_800['total_receipts_amount'] > 0]['total_receipts_amount'].sum()
        print(f"Avg (Reimbursement/Receipts) for receipts $600-$800: {avg_ratio_600_800:.2f}")
        findings.append(f"Avg (Reimbursement/Receipts) for receipts $600-$800: {avg_ratio_600_800:.2f}. Lisa suggested good treatment.")

    # Rounding bug: receipts ending in .49 or .99 (Lisa)
    cents_49 = df[df['receipt_cents'] == 49]
    cents_99 = df[df['receipt_cents'] == 99]
    other_cents = df[~df['receipt_cents'].isin([49, 99])]
    
    # Compare reimbursement deviation from a simple model (e.g. per_diem + mileage + receipts)
    # This is tricky without a baseline. For now, just compare average reimbursement or RPD.
    if not cents_49.empty:
        avg_rpd_49_cents = cents_49['reimbursement_per_day'].mean()
        print(f"Avg RPD for receipts ending in .49: ${avg_rpd_49_cents:.2f}")
        findings.append(f"Avg RPD for receipts ending in .49: ${avg_rpd_49_cents:.2f}.")
    if not cents_99.empty:
        avg_rpd_99_cents = cents_99['reimbursement_per_day'].mean()
        print(f"Avg RPD for receipts ending in .99: ${avg_rpd_99_cents:.2f}")
        findings.append(f"Avg RPD for receipts ending in .99: ${avg_rpd_99_cents:.2f}.")
    if not other_cents.empty:
        avg_rpd_other_cents = other_cents['reimbursement_per_day'].mean()
        print(f"Avg RPD for other receipt cents: ${avg_rpd_other_cents:.2f}")
        findings.append(f"Avg RPD for other receipt cents: ${avg_rpd_other_cents:.2f}.")
        if not cents_49.empty and avg_rpd_49_cents > avg_rpd_other_cents:
             findings.append("Receipts ending in .49 show higher RPD than others.")
        if not cents_99.empty and avg_rpd_99_cents > avg_rpd_other_cents:
             findings.append("Receipts ending in .99 show higher RPD than others.")


    # Kevin's optimal spending ranges based on trip length
    # Short trips (<4 days?), Medium (4-6 days), Long (>7 days?)
    # Define trip length categories
    df['trip_length_category'] = 'Medium' # Default
    df.loc[df['trip_duration_days'] < 4, 'trip_length_category'] = 'Short'
    df.loc[df['trip_duration_days'] > 6, 'trip_length_category'] = 'Long'

    print("\nKevin's spending/day hypotheses:")
    # Short trips: < $75/day optimal
    short_trips = df[df['trip_length_category'] == 'Short']
    if not short_trips.empty:
        st_optimal_spend = short_trips[short_trips['receipt_amount_per_day'] < 75]
        st_over_spend = short_trips[short_trips['receipt_amount_per_day'] >= 75]
        if not st_optimal_spend.empty and not st_over_spend.empty:
            print(f"Short Trips (<4d): Avg RPD for <$75/day receipts: ${st_optimal_spend['reimbursement_per_day'].mean():.2f}")
            print(f"Short Trips (<4d): Avg RPD for >=$75/day receipts: ${st_over_spend['reimbursement_per_day'].mean():.2f}")
            findings.append(f"Short Trips (<4d): Avg RPD for <$75/day receipts: ${st_optimal_spend['reimbursement_per_day'].mean():.2f} vs >=$75/day: ${st_over_spend['reimbursement_per_day'].mean():.2f}")

    # Medium trips (4-6 days): < $120/day optimal
    medium_trips = df[df['trip_length_category'] == 'Medium']
    if not medium_trips.empty:
        mt_optimal_spend = medium_trips[medium_trips['receipt_amount_per_day'] < 120]
        mt_over_spend = medium_trips[medium_trips['receipt_amount_per_day'] >= 120]
        if not mt_optimal_spend.empty and not mt_over_spend.empty:
            print(f"Medium Trips (4-6d): Avg RPD for <$120/day receipts: ${mt_optimal_spend['reimbursement_per_day'].mean():.2f}")
            print(f"Medium Trips (4-6d): Avg RPD for >=$120/day receipts: ${mt_over_spend['reimbursement_per_day'].mean():.2f}")
            findings.append(f"Medium Trips (4-6d): Avg RPD for <$120/day receipts: ${mt_optimal_spend['reimbursement_per_day'].mean():.2f} vs >=$120/day: ${mt_over_spend['reimbursement_per_day'].mean():.2f}")

    # Long trips (>6 days): < $90/day optimal
    long_trips = df[df['trip_length_category'] == 'Long']
    if not long_trips.empty:
        lt_optimal_spend = long_trips[long_trips['receipt_amount_per_day'] < 90]
        lt_over_spend = long_trips[long_trips['receipt_amount_per_day'] >= 90]
        if not lt_optimal_spend.empty and not lt_over_spend.empty:
            print(f"Long Trips (>6d): Avg RPD for <$90/day receipts: ${lt_optimal_spend['reimbursement_per_day'].mean():.2f}")
            print(f"Long Trips (>6d): Avg RPD for >=$90/day receipts: ${lt_over_spend['reimbursement_per_day'].mean():.2f}")
            findings.append(f"Long Trips (>6d): Avg RPD for <$90/day receipts: ${lt_optimal_spend['reimbursement_per_day'].mean():.2f} vs >=$90/day: ${lt_over_spend['reimbursement_per_day'].mean():.2f}")


def analyze_interactions_and_correlations(df, findings):
    """Looks for correlations and potential interactions."""
    print("\n--- Interactions and Correlations ---")
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()
    
    print("Correlation with 'expected_output':\n", correlation_matrix['expected_output'].sort_values(ascending=False))
    findings.append(f"Top 3 positive correlations with expected_output: {correlation_matrix['expected_output'].sort_values(ascending=False).index[1:4].tolist()}")


def attempt_component_estimation(df, findings):
    """Attempts to estimate base per diem and mileage rates."""
    print("\n--- Component Estimation (Preliminary) ---")
    # Lisa: "$100 a day seems to be the base [per diem]."
    # Try to find cases with low miles and low receipts
    low_miles_low_receipts = df[(df['miles_traveled'] < 50) & (df['total_receipts_amount'] < 50)]
    if not low_miles_low_receipts.empty:
        estimated_per_diem = (low_miles_low_receipts['expected_output'] / low_miles_low_receipts['trip_duration_days']).mean()
        print(f"Estimated per diem from low_miles_low_receipts cases: ${estimated_per_diem:.2f}/day")
        findings.append(f"Estimated per diem from low_miles_low_receipts cases: ${estimated_per_diem:.2f}/day (Lisa suggested $100).")

    # Try to estimate mileage rate from cases with 0 receipts
    zero_receipts = df[df['total_receipts_amount'] == 0]
    if not zero_receipts.empty and not zero_receipts[zero_receipts['miles_traveled'] > 0].empty:
        # Assuming per_diem_guess * trip_duration_days + mileage_rate * miles_traveled = expected_output
        # mileage_rate = (expected_output - per_diem_guess * trip_duration_days) / miles_traveled
        per_diem_guess = 100 # From Lisa
        zero_receipts_miles_positive = zero_receipts[zero_receipts['miles_traveled'] > 0].copy() # Use .copy()
        zero_receipts_miles_positive.loc[:, 'estimated_mileage_reimbursement'] = zero_receipts_miles_positive['expected_output'] - (per_diem_guess * zero_receipts_miles_positive['trip_duration_days'])
        zero_receipts_miles_positive.loc[:, 'estimated_mileage_rate'] = zero_receipts_miles_positive['estimated_mileage_reimbursement'] / zero_receipts_miles_positive['miles_traveled']
        
        avg_est_mileage_rate = zero_receipts_miles_positive[zero_receipts_miles_positive['estimated_mileage_rate'] > 0]['estimated_mileage_rate'].mean() # Filter out negative rates
        print(f"Estimated mileage rate from 0-receipt cases (assuming $100/day per diem): ${avg_est_mileage_rate:.3f}/mile")
        findings.append(f"Estimated mileage rate from 0-receipt cases (assuming $100/day per diem): ${avg_est_mileage_rate:.3f}/mile.")


def main():
    """Main function to run the analysis."""
    findings = [] # List to collect actionable findings
    
    df_raw = load_data()
    df = create_derived_features(df_raw)
    
    print("--- Basic Data Exploration ---")
    print("Shape of the DataFrame:", df.shape)
    print("\nDataFrame Info:")
    df.info()
    print("\nDescriptive Statistics (Original Inputs and Output):")
    print(df_raw.describe())
    print("\nDescriptive Statistics (Including Derived Features):")
    print(df.describe())
    
    # Plot distributions of key original and derived features
    plot_distributions(df, ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output'], "Original ")
    plot_distributions(df, ['miles_per_day', 'receipt_amount_per_day', 'reimbursement_per_day', 'reimbursement_per_mile'], "Derived ")
    
    analyze_trip_duration(df, findings)
    analyze_mileage(df, findings) # This also analyzes miles_per_day
    analyze_receipts(df, findings) # This also analyzes receipt_cents and Kevin's spending rules
    
    analyze_interactions_and_correlations(df, findings)
    attempt_component_estimation(df, findings)

    # Kevin: "Six different calculation paths" -> This needs more advanced clustering or decision tree analysis.
    # For now, suggest manual segmentation based on findings.
    findings.append("Consider segmenting analysis/rules by trip_duration_days (e.g., 1-2, 3-4, 5, 6-7, 8+ days) or miles_per_day categories, as interviewees suggest different behaviors.")

    print("\n\n--- Summary of Actionable Findings ---")
    if not findings:
        print("No specific actionable findings generated yet. Review plots and detailed stats.")
    for i, finding in enumerate(findings):
        print(f"{i+1}. {finding}")

if __name__ == '__main__':
    main()
    print("\nData analysis script finished. Review the plots and printed output for insights.")

