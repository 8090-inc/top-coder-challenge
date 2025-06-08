import json
import pandas as pd

def load_data(filepath):
    """Loads data from a JSON file and normalizes the nested 'input' fields."""
    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f) # This is a list of dicts

        # Create a DataFrame from the raw data
        # 'expected_output' will be a column, 'input' will be a column of dicts
        df_initial = pd.DataFrame(raw_data)

        if 'input' not in df_initial.columns:
            print("Error: 'input' column not found in the initial data structure.")
            return None

        # Normalize the 'input' column (which contains dictionaries)
        # This creates a new DataFrame from the dictionaries in the 'input' column
        input_data_normalized = pd.json_normalize(df_initial['input'])

        # Concatenate the normalized input data with the 'expected_output' column
        # Ensure 'expected_output' is present
        if 'expected_output' not in df_initial.columns:
            print("Error: 'expected_output' column not found in the initial data structure.")
            # Decide how to handle: maybe return df with only input cols, or None
            return input_data_normalized # Or handle as error

        df_final = pd.concat([input_data_normalized, df_initial[['expected_output']]], axis=1)

        return df_final
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def main():
    filepath = 'public_cases.json'
    df = load_data(filepath)

    if df is None:
        return

    print("Successfully loaded data.")
    # print(f"Columns in DataFrame: {df.columns.tolist()}") # Optionally keep for one more verification

    # Task 2: Calculate and print basic descriptive statistics
    print("\nDescriptive Statistics:")
    cols_to_describe = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output']
    for col in cols_to_describe:
        if col in df.columns:
            print(f"\nStatistics for {col}:")
            print(df[col].describe())
        else:
            print(f"\nWarning: Column '{col}' not found in the DataFrame.")

    # Task 3: Group data by trip_duration_days
    if 'trip_duration_days' in df.columns and 'expected_output' in df.columns:
        print("\nAverage expected_output by trip_duration_days:")
        avg_output_by_duration = df.groupby('trip_duration_days')['expected_output'].mean()
        print(avg_output_by_duration)

        # Specifically note any obvious bonus or penalty for 5-day trips
        if 5 in avg_output_by_duration.index:
            avg_5_day = avg_output_by_duration.loc[5]
            bonus_penalty_info = f"Average for 5-day trips: {avg_5_day:.2f}\n"
            if 4 in avg_output_by_duration.index:
                avg_4_day = avg_output_by_duration.loc[4]
                bonus_penalty_info += f"  Compared to 4-day trips (avg: {avg_4_day:.2f}): "
                if avg_5_day > avg_4_day:
                    bonus_penalty_info += f"Higher (bonus of {avg_5_day - avg_4_day:.2f})\n"
                elif avg_5_day < avg_4_day:
                    bonus_penalty_info += f"Lower (penalty of {avg_4_day - avg_5_day:.2f})\n"
                else:
                    bonus_penalty_info += "Similar\n"
            if 6 in avg_output_by_duration.index:
                avg_6_day = avg_output_by_duration.loc[6]
                bonus_penalty_info += f"  Compared to 6-day trips (avg: {avg_6_day:.2f}): "
                if avg_5_day > avg_6_day:
                    bonus_penalty_info += f"Higher (bonus of {avg_5_day - avg_6_day:.2f})"
                elif avg_5_day < avg_6_day:
                    bonus_penalty_info += f"Lower (penalty of {avg_6_day - avg_5_day:.2f})"
                else:
                    bonus_penalty_info += "Similar"
            print(f"\nBonus/Penalty analysis for 5-day trips:\n{bonus_penalty_info}")
        else:
            print("\nNo 5-day trips found to analyze for bonus/penalty.")
    else:
        print("\nSkipping Task 3: 'trip_duration_days' or 'expected_output' column not found.")

    # Task 4: Analyze mileage impact
    if 'trip_duration_days' in df.columns and 'miles_traveled' in df.columns and 'expected_output' in df.columns:
        print("\nAnalyzing mileage impact on expected_output:")
        # Find a common trip duration. Let's check value counts.
        common_duration = df['trip_duration_days'].mode()
        if not common_duration.empty:
            common_duration = common_duration[0]
            print(f"Focusing on most common trip duration: {common_duration} days")
            df_common_duration = df[df['trip_duration_days'] == common_duration].copy() # Use .copy() to avoid SettingWithCopyWarning

            if not df_common_duration.empty:
                # Create mileage bins
                try:
                    df_common_duration['mileage_bins'] = pd.qcut(df_common_duration['miles_traveled'], q=4, duplicates='drop')
                    avg_output_by_mileage_bin = df_common_duration.groupby('mileage_bins')['expected_output'].mean()
                    print(f"\nAverage expected_output by mileage bins for {common_duration}-day trips:")
                    print(avg_output_by_mileage_bin)

                    correlation = df_common_duration['miles_traveled'].corr(df_common_duration['expected_output'])
                    print(f"\nCorrelation between miles_traveled and expected_output for {common_duration}-day trips: {correlation:.2f}")
                    if correlation > 0.5:
                        print("This suggests a strong positive relationship: as mileage increases, expected output tends to increase.")
                    elif correlation > 0.1:
                        print("This suggests a weak to moderate positive relationship.")
                    elif correlation < -0.5:
                        print("This suggests a strong negative relationship.")
                    elif correlation < -0.1:
                        print("This suggests a weak to moderate negative relationship.")
                    else:
                        print("This suggests a very weak or no linear relationship.")
                except ValueError as e:
                    print(f"Could not create mileage bins, possibly due to too few data points or identical values: {e}")
                    # Fallback: print correlation if binning fails but data exists
                    if len(df_common_duration) > 1:
                        correlation = df_common_duration['miles_traveled'].corr(df_common_duration['expected_output'])
                        print(f"\nCorrelation between miles_traveled and expected_output for {common_duration}-day trips: {correlation:.2f}")
                    else:
                        print("Not enough data to calculate correlation for mileage impact.")

            else:
                print(f"No data found for the most common duration ({common_duration} days) to analyze mileage impact.")
        else:
            print("Could not determine a common trip duration for mileage impact analysis.")
    else:
        print("\nSkipping Task 4: One or more required columns ('trip_duration_days', 'miles_traveled', 'expected_output') not found.")

    # Task 5: Analyze receipt impact
    if 'total_receipts_amount' in df.columns and 'expected_output' in df.columns:
        print("\nAnalyzing receipt impact on expected_output:")

        # Part 1: Cases with total_receipts_amount < $20
        low_receipt_cases = df[df['total_receipts_amount'] < 20][['total_receipts_amount', 'expected_output']]
        if not low_receipt_cases.empty:
            print("\nExamples of cases with total_receipts_amount < $20:")
            print(low_receipt_cases.head()) # Print first 5 examples
        else:
            print("\nNo cases found with total_receipts_amount < $20.")

        # Part 2: Cases where total_receipts_amount ends with .49 or .99
        # Ensure total_receipts_amount is float, handle potential NAs
        df['total_receipts_amount'] = pd.to_numeric(df['total_receipts_amount'], errors='coerce')
        df_receipt_analysis = df.dropna(subset=['total_receipts_amount'])

        # Check for .49 or .99 by examining the fractional part
        # Multiply by 100, take modulo 100, and check if it's close to 49 or 99 (due to float precision)
        ends_with_49_or_99 = df_receipt_analysis['total_receipts_amount'].apply(
            lambda x: abs(round((x * 100) % 100) - 49) < 0.01 or abs(round((x * 100) % 100) - 99) < 0.01
        )
        cases_49_99 = df_receipt_analysis[ends_with_49_or_99]

        num_cases_49_99 = len(cases_49_99)
        print(f"\nNumber of cases where total_receipts_amount ends with .49 or .99: {num_cases_49_99}")

        if num_cases_49_99 > 0:
            avg_output_49_99 = cases_49_99['expected_output'].mean()
            print(f"Average expected_output for these cases: {avg_output_49_99:.2f}")

            other_cases = df_receipt_analysis[~ends_with_49_or_99]
            if not other_cases.empty:
                 avg_output_others = other_cases['expected_output'].mean()
                 print(f"Average expected_output for other cases (receipts not ending in .49 or .99): {avg_output_others:.2f}")
            else:
                print("No other cases to compare against for .49/.99 analysis.")
        else:
            print("No cases found where total_receipts_amount ends with .49 or .99.")
    else:
        print("\nSkipping Task 5: 'total_receipts_amount' or 'expected_output' column not found.")

    # Task 6: Calculate and analyze efficiency
    if 'miles_traveled' in df.columns and 'trip_duration_days' in df.columns and 'expected_output' in df.columns:
        print("\nAnalyzing efficiency (miles_traveled / trip_duration_days):")

        df_efficiency = df.copy()
        df_efficiency['trip_duration_days_numeric'] = pd.to_numeric(df_efficiency['trip_duration_days'], errors='coerce')

        valid_duration_mask = df_efficiency['trip_duration_days_numeric'].notna() & (df_efficiency['trip_duration_days_numeric'] > 0)
        df_efficiency_valid = df_efficiency[valid_duration_mask].copy() # Use .copy() here

        if not df_efficiency_valid.empty:
            df_efficiency_valid.loc[:, 'efficiency'] = df_efficiency_valid['miles_traveled'] / df_efficiency_valid['trip_duration_days_numeric']

            print("\nDescriptive statistics for efficiency:")
            print(df_efficiency_valid['efficiency'].describe())

            common_duration_eff = df_efficiency_valid['trip_duration_days_numeric'].mode()
            if not common_duration_eff.empty:
                common_duration_eff = common_duration_eff[0]
                df_common_duration_eff = df_efficiency_valid[df_efficiency_valid['trip_duration_days_numeric'] == common_duration_eff]

                if len(df_common_duration_eff) > 1:
                    low_eff_threshold = df_common_duration_eff['efficiency'].quantile(0.10)
                    high_eff_threshold = df_common_duration_eff['efficiency'].quantile(0.90)

                    very_low_eff_cases = df_common_duration_eff[df_common_duration_eff['efficiency'] <= low_eff_threshold]
                    very_high_eff_cases = df_common_duration_eff[df_common_duration_eff['efficiency'] >= high_eff_threshold]

                    avg_output_overall_common_duration = df_common_duration_eff['expected_output'].mean()
                    print(f"\nFor {common_duration_eff}-day trips (avg expected_output: {avg_output_overall_common_duration:.2f}):")

                    if not very_low_eff_cases.empty:
                        avg_output_low_eff = very_low_eff_cases['expected_output'].mean()
                        print(f"  Avg expected_output for very low efficiency (<= {low_eff_threshold:.2f} miles/day): {avg_output_low_eff:.2f}")
                    else:
                        print(f"  No cases found with very low efficiency for {common_duration_eff}-day trips.")

                    if not very_high_eff_cases.empty:
                        avg_output_high_eff = very_high_eff_cases['expected_output'].mean()
                        print(f"  Avg expected_output for very high efficiency (>= {high_eff_threshold:.2f} miles/day): {avg_output_high_eff:.2f}")
                    else:
                        print(f"  No cases found with very high efficiency for {common_duration_eff}-day trips.")
                else:
                    print(f"Not enough data for {common_duration_eff}-day trips to analyze high/low efficiency impact.")
            else:
                print("Could not determine a common trip duration for efficiency impact analysis.")
        else:
            print("No valid data (trip_duration_days > 0) to calculate efficiency.")

    else:
        print("\nSkipping Task 6: 'miles_traveled', 'trip_duration_days', or 'expected_output' column not found.")

    # Task 7: Summarize key observations
    print("\n" + "="*30 + " KEY OBSERVATIONS SUMMARY " + "="*30)
    print("1. Trip Duration:")
    print("   - Average 'expected_output' generally increases with trip duration (e.g., 1-day avg: 873.55, 14-day avg: 1707.07).")
    print("   - For 5-day trips (avg: 1272.59):")
    print("     - Higher than 4-day trips (avg: 1217.96, bonus of 54.63).")
    print("     - Lower than 6-day trips (avg: 1366.48, penalty of 93.89). This suggests a non-linear relationship around this duration.")
    print("2. Mileage Impact (focused on 5-day trips):")
    print("   - There's a weak to moderate positive correlation (0.49) between 'miles_traveled' and 'expected_output'.")
    print("   - Average 'expected_output' increases with mileage bins: from ~972 for the lowest quartile of miles to ~1478 for the highest.")
    print("3. Receipt Impact:")
    print("   - Cases with very low 'total_receipts_amount' (< $20) have significantly lower 'expected_output' (e.g., examples shown are mostly below $400).")
    print("   - The 30 cases where 'total_receipts_amount' ends in .49 or .99 have a drastically lower average 'expected_output' (566.19) compared to other cases (1373.33). This is a strong signal and might indicate specific types of expenses or reporting behavior.")
    print("4. Efficiency (Miles/Day):")
    print("   - Efficiency has a wide distribution (mean: 147, std: 193, min: 0.5, max: 1166).")
    print("   - For 5-day trips (most common duration, avg expected_output: 1272.59):")
    print("     - Very low efficiency (<= 25.03 miles/day) correlates with lower 'expected_output' (avg: 902.20).")
    print("     - Very high efficiency (>= 202.72 miles/day) correlates with higher 'expected_output' (avg: 1541.20).")
    print("5. Data Quality/Other:")
    print("   - Data was successfully loaded and key numeric features (duration, mileage, receipts, output) are available for all 1000 records.")
    print("   - No widespread missing values were explicitly noted for the core features analyzed after normalization.")
    print("   - The strong impact of receipt amounts ending in .49/.99 is a particularly striking pattern worth further investigation for feature engineering or rule-based adjustments.")
    print("="*78)

if __name__ == '__main__':
    main()
