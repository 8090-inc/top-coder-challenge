import json
import pandas as pd

def analyze_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract relevant fields into a list of dictionaries
    processed_data = []
    for item in data:
        record = {
            'trip_duration_days': item['input']['trip_duration_days'],
            'miles_traveled': item['input']['miles_traveled'],
            'total_receipts_amount': item['input']['total_receipts_amount'],
            'expected_output': item['expected_output']
        }
        processed_data.append(record)

    # Create a Pandas DataFrame
    df = pd.DataFrame(processed_data)

    # Calculate summary statistics
    summary_stats = df.describe().loc[['min', 'max', 'mean', '50%', 'std']]
    summary_stats = summary_stats.rename(index={'50%': 'median'})

    # Print summary statistics (will be captured by the tool output)
    print("Summary Statistics:\n")
    print("Trip Duration (days):\n", summary_stats['trip_duration_days'])
    print("\nંત્રtool_code\nMiles Traveled:\n", summary_stats['miles_traveled'])
    print("\nTotal Receipts Amount:\n", summary_stats['total_receipts_amount'])
    print("\nExpected Output:\n", summary_stats['expected_output'])

    # Brief observations on potential correlations
    print("\nPotential Correlations:\n")
    # Correlation with expected_output
    correlations = df.corr()['expected_output'].sort_values(ascending=False)
    print(correlations)

    # General observations (add more based on visual inspection or further analysis if needed)
    if correlations['trip_duration_days'] > 0.5:
        print("\n- Expected output generally increases with trip_duration_days.")
    elif correlations['trip_duration_days'] < -0.5:
        print("\n- Expected output generally decreases with trip_duration_days.")
    else:
        print("\n- Correlation between expected_output and trip_duration_days is moderate or weak.")

    if correlations['miles_traveled'] > 0.5:
        print("\n- Expected output generally increases with miles_traveled.")
    elif correlations['miles_traveled'] < -0.5:
        print("\n- Correlation between expected_output and miles_traveled is moderate or weak.")
    else: # Check if it's between -0.5 and 0.5 but not exactly 0 for a more nuanced message
        if -0.5 <= correlations['miles_traveled'] <= 0.5 and correlations['miles_traveled'] != 0:
             print("\n- Correlation between expected_output and miles_traveled is moderate or weak.")
        else: # if it's 0 or very close
             print("\n- No significant linear correlation between expected_output and miles_traveled.")


    if correlations['total_receipts_amount'] > 0.5:
        print("\n- Expected output generally increases with total_receipts_amount.")
    elif correlations['total_receipts_amount'] < -0.5:
        print("\n- Expected output generally decreases with total_receipts_amount.")
    else:
        print("\n- Correlation between expected_output and total_receipts_amount is moderate or weak.")

if __name__ == "__main__":
    analyze_json_data("public_cases.json")
