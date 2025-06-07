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
warnings.filterwarnings('ignore', category=FutureWarning) # For pandas an upcoming changes

MIN_SEGMENT_SIZE = 20 # Minimum number of data points to attempt regression

# --- Helper Functions ---
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

def create_derived_features(df_in):
    df = df_in.copy()
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days'].replace(0, np.nan)
    df['receipt_amount_per_day'] = df['total_receipts_amount'] / df['trip_duration_days'].replace(0, np.nan)
    
    # Baseline linear model from previous analysis
    df['predicted_global_linear'] = 266.71 + \
                                   (0.38 * df['total_receipts_amount']) + \
                                   (50.05 * df['trip_duration_days']) + \
                                   (0.45 * df['miles_traveled'])
    df['residual_global_linear'] = df['expected_output'] - df['predicted_global_linear']
    return df

def fit_segment_model(segment_df, segment_name, findings_list):
    """Fits a simple linear model to a segment and reports findings."""
    if len(segment_df) < MIN_SEGMENT_SIZE:
        findings_list.append(f"Segment '{segment_name}': Too small ({len(segment_df)} data points). Skipping detailed model.")
        return None

    X_cols = ['total_receipts_amount', 'trip_duration_days', 'miles_traveled']
    X = segment_df[X_cols].values
    # Add intercept column of ones
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    y = segment_df['expected_output'].values

    try:
        coeffs, residuals_sum_sq, rank, singular_values = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        intercept, coeff_receipts, coeff_days, coeff_miles = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        
        predicted_segment = X_with_intercept @ coeffs
        mae_segment = np.mean(np.abs(y - predicted_segment))
        
        finding = (f"Segment '{segment_name}' (Size: {len(segment_df)}):\n"
                   f"  Formula: E_Out = {intercept:.2f} + ({coeff_receipts:.2f}*Receipts) + ({coeff_days:.2f}*Days) + ({coeff_miles:.2f}*Miles)\n"
                   f"  MAE within segment: ${mae_segment:.2f}")
        findings_list.append(finding)
        
        # Compare with global model's MAE for this segment
        mae_global_on_segment = np.mean(np.abs(segment_df['residual_global_linear']))
        findings_list.append(f"  Global Model MAE on this segment: ${mae_global_on_segment:.2f}")
        if mae_segment < mae_global_on_segment:
             findings_list.append(f"  Segment model is BETTER than global for this segment.")
        
        return {'name': segment_name, 'size': len(segment_df), 'intercept': intercept, 
                'coeff_receipts': coeff_receipts, 'coeff_days': coeff_days, 
                'coeff_miles': coeff_miles, 'mae': mae_segment}
                
    except np.linalg.LinAlgError as e:
        findings_list.append(f"Segment '{segment_name}': Linear regression failed. Error: {e}")
        return None

def define_segments(df):
    """Defines various segmentation schemes and adds them as columns to the DataFrame."""
    df_segmented = df.copy()

    # 1. Trip Duration Segmentation
    df_segmented['seg_duration'] = 'd_unknown'
    df_segmented.loc[df_segmented['trip_duration_days'] == 1, 'seg_duration'] = 'd_1'
    df_segmented.loc[df_segmented['trip_duration_days'].between(2, 3), 'seg_duration'] = 'd_2_3'
    df_segmented.loc[df_segmented['trip_duration_days'].between(4, 6), 'seg_duration'] = 'd_4_6'
    df_segmented.loc[df_segmented['trip_duration_days'] >= 7, 'seg_duration'] = 'd_7_plus'

    # 2. Mileage Segmentation
    df_segmented['seg_mileage'] = 'm_unknown'
    df_segmented.loc[df_segmented['miles_traveled'] == 0, 'seg_mileage'] = 'm_zero'
    df_segmented.loc[(df_segmented['miles_traveled'] > 0) & (df_segmented['miles_traveled'] <= 100), 'seg_mileage'] = 'm_short_0_100'
    df_segmented.loc[(df_segmented['miles_traveled'] > 100) & (df_segmented['miles_traveled'] <= 500), 'seg_mileage'] = 'm_medium_101_500'
    df_segmented.loc[df_segmented['miles_traveled'] > 500, 'seg_mileage'] = 'm_long_501_plus'

    # 3. Total Receipt Amount Segmentation (using fixed bands for interpretability)
    df_segmented['seg_receipt_total'] = 'r_unknown'
    df_segmented.loc[df_segmented['total_receipts_amount'] <= 200, 'seg_receipt_total'] = 'r_low_0_200'
    df_segmented.loc[(df_segmented['total_receipts_amount'] > 200) & (df_segmented['total_receipts_amount'] <= 1000), 'seg_receipt_total'] = 'r_medium_201_1000'
    df_segmented.loc[df_segmented['total_receipts_amount'] > 1000, 'seg_receipt_total'] = 'r_high_1001_plus'
    
    # 4. Receipt Amount Per Day Segmentation
    df_segmented['seg_receipt_per_day'] = 'rpd_unknown'
    # Handle cases where trip_duration_days might be 0 if data is weird, though problem implies it's >= 1
    # For receipt_amount_per_day, NaNs might exist if trip_duration_days was 0. Fill with a placeholder or handle.
    df_segmented.loc[df_segmented['receipt_amount_per_day'] <= 50, 'seg_receipt_per_day'] = 'rpd_low_0_50'
    df_segmented.loc[(df_segmented['receipt_amount_per_day'] > 50) & (df_segmented['receipt_amount_per_day'] <= 150), 'seg_receipt_per_day'] = 'rpd_medium_51_150'
    df_segmented.loc[df_segmented['receipt_amount_per_day'] > 150, 'seg_receipt_per_day'] = 'rpd_high_151_plus'
    df_segmented['seg_receipt_per_day'] = df_segmented['seg_receipt_per_day'].fillna('rpd_unknown')


    # 5. Miles Per Day (Efficiency) Segmentation
    df_segmented['seg_miles_per_day'] = 'mpd_unknown'
    df_segmented.loc[df_segmented['miles_per_day'] < 50, 'seg_miles_per_day'] = 'mpd_low_0_49'
    df_segmented.loc[(df_segmented['miles_per_day'] >= 50) & (df_segmented['miles_per_day'] < 150), 'seg_miles_per_day'] = 'mpd_medium_50_149'
    df_segmented.loc[df_segmented['miles_per_day'] >= 150, 'seg_miles_per_day'] = 'mpd_high_150_plus'
    df_segmented['seg_miles_per_day'] = df_segmented['seg_miles_per_day'].fillna('mpd_unknown') # Handle NaNs from miles_per_day if days=0

    # 6. Global Model Residual Segmentation
    df_segmented['seg_residual'] = 'resid_mid'
    df_segmented.loc[df_segmented['residual_global_linear'] < -200, 'seg_residual'] = 'resid_large_neg' # Global model overpaid
    df_segmented.loc[df_segmented['residual_global_linear'] > 200, 'seg_residual'] = 'resid_large_pos'  # Global model underpaid
    
    return df_segmented

def main():
    findings_summary = []
    all_segment_models = []

    df_loaded = load_data()
    df_with_features = create_derived_features(df_loaded)
    df_segmented = define_segments(df_with_features)

    print("--- Segmentation Analysis ---")
    findings_summary.append("Segmentation Analysis Results:")

    segment_columns = ['seg_duration', 'seg_mileage', 'seg_receipt_total', 'seg_receipt_per_day', 'seg_miles_per_day', 'seg_residual']

    # Analyze primary segments
    for seg_col in segment_columns:
        findings_summary.append(f"\n--- Analyzing by {seg_col} ---")
        print(f"\n--- Analyzing by {seg_col} ---")
        for segment_value in df_segmented[seg_col].unique():
            current_segment_df = df_segmented[df_segmented[seg_col] == segment_value]
            model_info = fit_segment_model(current_segment_df, f"{seg_col}_{segment_value}", findings_summary)
            if model_info:
                all_segment_models.append(model_info)
    
    # Analyze selected 2-level combined segments (examples)
    # Focus on combinations that seem promising from primary analysis or previous findings
    print("\n--- Analyzing Selected Combined Segments ---")
    findings_summary.append("\n--- Analyzing Selected Combined Segments ---")

    # Example 1: 1-day trips with short mileage
    seg_1d_short_miles_df = df_segmented[(df_segmented['seg_duration'] == 'd_1') & (df_segmented['seg_mileage'] == 'm_short_0_100')]
    model_info = fit_segment_model(seg_1d_short_miles_df, "d_1_AND_m_short_0_100", findings_summary)
    if model_info: all_segment_models.append(model_info)

    # Example 2: Long duration trips with high total receipts
    seg_long_d_high_r_df = df_segmented[(df_segmented['seg_duration'] == 'd_7_plus') & (df_segmented['seg_receipt_total'] == 'r_high_1001_plus')]
    model_info = fit_segment_model(seg_long_d_high_r_df, "d_7_plus_AND_r_high_1001_plus", findings_summary)
    if model_info: all_segment_models.append(model_info)
    
    # Example 3: Trips where global linear model overpaid significantly (resid_large_neg) AND had high receipts
    seg_resid_neg_high_r_df = df_segmented[(df_segmented['seg_residual'] == 'resid_large_neg') & (df_segmented['seg_receipt_total'] == 'r_high_1001_plus')]
    model_info = fit_segment_model(seg_resid_neg_high_r_df, "resid_large_neg_AND_r_high_1001_plus", findings_summary)
    if model_info: all_segment_models.append(model_info)

    # Example 4: Trips with high miles_per_day
    seg_high_mpd_df = df_segmented[df_segmented['seg_miles_per_day'] == 'mpd_high_150_plus']
    model_info = fit_segment_model(seg_high_mpd_df, "mpd_high_150_plus", findings_summary) # This is also a primary one, but good to re-check context
    if model_info and not any(m['name'] == "mpd_high_150_plus" for m in all_segment_models): # Avoid duplicate if already added
         all_segment_models.append(model_info)


    print("\n\n--- Summary of All Segmentation Findings & Potential Paths ---")
    # Sort models by how much better they are than global, or by MAE
    all_segment_models_df = pd.DataFrame(all_segment_models)
    if not all_segment_models_df.empty:
        # Add global MAE for comparison if possible (needs to be calculated or passed)
        # For now, just print what we have
        print(all_segment_models_df.sort_values(by='mae', ascending=True))
    
    for finding in findings_summary:
        print(finding)
    
    # df_segmented.to_csv("segmentation_analysis_output.csv", index=False)
    # print("\nSaved detailed segmented data to segmentation_analysis_output.csv")

    print("\n--- End of Segmentation Analysis Script ---")
    print("Review the MAE for each segment. Segments with low MAE and distinct coefficients might represent a unique calculation path.")
    print("Look for 4-6 segments or groups of segments that cover most data with good local formulas.")

if __name__ == '__main__':
    main()
