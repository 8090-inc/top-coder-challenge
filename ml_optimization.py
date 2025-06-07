import json
import pandas as pd
import numpy as np
import warnings

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # For XGBoost verbosity if not configured

# --- Configuration & Constants ---
RANDOM_STATE = 42
N_ESTIMATORS_RF = 150 # Increased for potentially more complex patterns
MAX_DEPTH_RF = 12     # Allow deeper trees for RF
MIN_SAMPLES_SPLIT_RF = 10
MIN_SAMPLES_LEAF_RF = 5

N_ESTIMATORS_XGB = 200 # Increased
MAX_DEPTH_XGB = 7      # Allow deeper trees for XGB
LEARNING_RATE_XGB = 0.05
SUBSAMPLE_XGB = 0.8
COLSAMPLE_BYTREE_XGB = 0.8


MAX_DEPTH_DT_RULES = 4 # For extracting simple rules
MIN_SAMPLES_LEAF_DT_RULES = 30 # Increased to get more general rules

# --- Helper Functions ---
def load_data(file_path="public_cases.json"):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Make sure 'public_cases.json' is in the same directory as the script.")
        return pd.DataFrame() # Return empty DataFrame

    inputs = [item['input'] for item in data]
    outputs = [item['expected_output'] for item in data]
    df = pd.DataFrame(inputs)
    df['expected_output'] = outputs
    
    # Ensure correct data types and handle potential errors during conversion
    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where essential numeric conversion failed or essential data is missing
    df.dropna(subset=['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output'], inplace=True)
    df['trip_duration_days'] = df['trip_duration_days'].astype(int)
    return df

def advanced_feature_engineering(df_in):
    df = df_in.copy()
    if df.empty:
        return df

    # Basic Ratios (handle division by zero)
    df['miles_per_day'] = np.where(df['trip_duration_days'] > 0, df['miles_traveled'] / df['trip_duration_days'], 0)
    df['receipt_amount_per_day'] = np.where(df['trip_duration_days'] > 0, df['total_receipts_amount'] / df['trip_duration_days'], 0)

    # Trip Duration Features
    df['is_1_day_trip'] = (df['trip_duration_days'] == 1).astype(int)
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)
    duration_bins = [0, 1, 3, 6, 10, df['trip_duration_days'].max() + 1] # Ensure max is covered
    duration_labels = ['1d', '2-3d', '4-6d', '7-10d', f"11-{df['trip_duration_days'].max()}d"]
    if df['trip_duration_days'].max() < 11: # Adjust labels if max duration is less than 11
        duration_bins = [0, 1, 3, 6, df['trip_duration_days'].max() + 1]
        duration_labels = ['1d', '2-3d', '4-6d', f"7-{df['trip_duration_days'].max()}d"]
    df['duration_cat_obj'] = pd.cut(df['trip_duration_days'],
                                bins=duration_bins,
                                labels=duration_labels[:len(duration_bins)-1], # Match labels to bins
                                right=True, include_lowest=True)

    # Miles Per Day (MPD) Efficiency
    df['mpd_sweet_spot_180_220'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    df['mpd_low_lt_50'] = (df['miles_per_day'] < 50).astype(int)
    df['mpd_high_gt_250'] = (df['miles_per_day'] > 250).astype(int)
    df['mpd_zero'] = (df['miles_per_day'] == 0).astype(int)


    # Receipt Tiers (from previous segmentation, can be refined by ML)
    df['receipt_tier_obj'] = 'medium_201_1000' # Default
    df.loc[df['total_receipts_amount'] <= 200, 'receipt_tier_obj'] = 'low_le_200'
    df.loc[df['total_receipts_amount'] > 1000, 'receipt_tier_obj'] = 'high_gt_1000'
    
    # Receipt Amount Per Day Tiers
    df['rapd_tier_obj'] = 'medium_51_150' # Default
    df.loc[df['receipt_amount_per_day'] <= 50, 'rapd_tier_obj'] = 'low_le_50'
    df.loc[df['receipt_amount_per_day'] > 150, 'rapd_tier_obj'] = 'high_gt_150'


    # Interaction Features
    df['days_x_miles'] = df['trip_duration_days'] * df['miles_traveled']
    df['days_x_receipts'] = df['trip_duration_days'] * df['total_receipts_amount']
    df['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    df['days_x_miles_per_day'] = df['trip_duration_days'] * df['miles_per_day'] # Effectively miles_traveled
    df['days_x_receipt_amount_per_day'] = df['trip_duration_days'] * df['receipt_amount_per_day'] # Effectively total_receipts_amount

    # Polynomial features for key variables
    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'receipt_amount_per_day']:
        if col in df.columns: 
            df[f'{col}_sq'] = df[col]**2
            # df[f'{col}_cub'] = df[col]**3 # Cube can sometimes overfit, start with square

    # One-hot encode categorical features created as objects
    categorical_cols_to_encode = []
    if 'duration_cat_obj' in df.columns: categorical_cols_to_encode.append('duration_cat_obj')
    if 'receipt_tier_obj' in df.columns: categorical_cols_to_encode.append('receipt_tier_obj')
    if 'rapd_tier_obj' in df.columns: categorical_cols_to_encode.append('rapd_tier_obj')
    
    # Add any other columns that are of 'object' or 'category' dtype and need encoding
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in categorical_cols_to_encode:
             categorical_cols_to_encode.append(col)
    
    df = pd.get_dummies(df, columns=[col for col in categorical_cols_to_encode if col in df.columns], 
                        prefix=[col.replace('_obj','_cat') for col in categorical_cols_to_encode if col in df.columns])
    
    # Ensure all engineered features are numeric and handle potential NaNs/Infs
    # This is crucial before passing to ML models
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaNs with 0
        df[col] = np.where(np.isinf(df[col]), 0, df[col]) # Replace Infs with 0

    return df

def train_random_forest(X_train, y_train, X_test, y_test, features_list, findings_list):
    print("\n--- Training Random Forest Regressor ---")
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE, n_jobs=-1, 
                               max_depth=MAX_DEPTH_RF, min_samples_split=MIN_SAMPLES_SPLIT_RF, 
                               min_samples_leaf=MIN_SAMPLES_LEAF_RF, oob_score=True)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    findings_list.append(f"Random Forest MAE on test set: ${mae_rf:.2f} (OOB Score: {rf.oob_score_:.4f})")
    print(f"Random Forest MAE on test set: ${mae_rf:.2f} (OOB Score: {rf.oob_score_:.4f})")

    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features_list, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    findings_list.append("Random Forest Top 15 Feature Importances:")
    print("Random Forest Top 15 Feature Importances:")
    for i, row in feature_importance_df.head(15).iterrows():
        findings_list.append(f"  - {row['feature']}: {row['importance']:.4f}")
        print(f"  - {row['feature']}: {row['importance']:.4f}")
    
    findings_list.append("\nExample Decision Tree Rules (interpret with caution, from shallow trees):")
    print("\nExample Decision Tree Rules (interpret with caution, from shallow trees):")
    try:
        # Fit a single shallow tree on the full training data for rule extraction
        dt_rules_model = DecisionTreeRegressor(max_depth=MAX_DEPTH_DT_RULES, 
                                               min_samples_leaf=MIN_SAMPLES_LEAF_DT_RULES, 
                                               random_state=RANDOM_STATE)
        dt_rules_model.fit(X_train, y_train) 
        rules = export_text(dt_rules_model, feature_names=features_list, show_weights=False, decimals=2) # show_weights=True can be very verbose
        findings_list.append(f"\n--- Shallow Decision Tree (depth {MAX_DEPTH_DT_RULES}) Rules ---")
        findings_list.append(rules)
        print(f"\n--- Shallow Decision Tree (depth {MAX_DEPTH_DT_RULES}) Rules ---")
        print(rules)
    except Exception as e:
        findings_list.append(f"Could not extract rules from shallow Decision Tree: {e}")
        print(f"Could not extract rules from shallow Decision Tree: {e}")
    return rf, mae_rf

def train_xgboost(X_train, y_train, X_test, y_test, features_list, findings_list):
    print("\n--- Training XGBoost Regressor ---")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=N_ESTIMATORS_XGB, random_state=RANDOM_STATE, 
                                 n_jobs=-1, learning_rate=LEARNING_RATE_XGB, max_depth=MAX_DEPTH_XGB, 
                                 subsample=SUBSAMPLE_XGB, colsample_bytree=COLSAMPLE_BYTREE_XGB,
                                 early_stopping_rounds=10) # Added early stopping
    
    # Use a validation set for early stopping
    eval_set = [(X_test, y_test)]
    xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    y_pred_xgb = xgb_model.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    findings_list.append(f"XGBoost MAE on test set: ${mae_xgb:.2f} (Best iteration: {xgb_model.best_iteration})")
    print(f"XGBoost MAE on test set: ${mae_xgb:.2f} (Best iteration: {xgb_model.best_iteration})")

    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features_list, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    findings_list.append("XGBoost Top 15 Feature Importances:")
    print("XGBoost Top 15 Feature Importances:")
    for i, row in feature_importance_df.head(15).iterrows():
        findings_list.append(f"  - {row['feature']}: {row['importance']:.4f}")
        print(f"  - {row['feature']}: {row['importance']:.4f}")
    return xgb_model, mae_xgb

def analyze_model_errors(model, X_data, y_data, df_original_inputs_with_derived, model_name, findings_list, num_cases=5):
    print(f"\n--- Error Analysis for {model_name} ---")
    # Ensure X_data has features in the same order as model was trained on
    if hasattr(model, 'feature_names_in_'):
        X_data_ordered = X_data[model.feature_names_in_]
    else: # For older scikit-learn or if not available
        X_data_ordered = X_data

    predictions = model.predict(X_data_ordered)
    errors = y_data - predictions
    abs_errors = np.abs(errors)
    
    # Create error_df using the index from X_data to align with df_original_inputs_with_derived
    error_df = df_original_inputs_with_derived.loc[X_data.index].copy()
    error_df['predicted'] = predictions
    error_df['expected'] = y_data.loc[X_data.index] # Align y_data as well
    error_df['error'] = errors.loc[X_data.index]
    error_df['abs_error'] = abs_errors.loc[X_data.index]
    
    worst_error_cases = error_df.sort_values(by='abs_error', ascending=False).head(num_cases)
    
    findings_list.append(f"Top {num_cases} worst error cases for {model_name}:")
    print(f"Top {num_cases} worst error cases for {model_name}:")
    for idx, row in worst_error_cases.iterrows():
        # Safely access columns, provide default if a derived feature for context is missing
        days = row.get('trip_duration_days', 'N/A')
        miles = row.get('miles_traveled', 'N/A')
        receipts_val = row.get('total_receipts_amount', 'N/A')
        
        receipt_str = "N/A"
        if isinstance(receipts_val, (int, float)):
            receipt_str = f"{receipts_val:.2f}"

        finding = (f"  Case Index {idx}: Inputs(days:{days}, miles:{miles}, receipts:${receipt_str}) -> "
                   f"Expected:${row['expected']:.2f}, Predicted:${row['predicted']:.2f}, Error:${row['error']:.2f}")
        findings_list.append(finding)
        print(finding)
    
    high_error_threshold = np.percentile(abs_errors, 90) 
    df_high_errors = error_df[error_df['abs_error'] >= high_error_threshold]
    if not df_high_errors.empty:
        findings_list.append(f"\nCharacteristics of top 10% error cases for {model_name} (abs_error > ${high_error_threshold:.2f}):")
        for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'receipt_amount_per_day']:
            if col in df_high_errors.columns:
                mean_val = df_high_errors[col].mean()
                median_val = df_high_errors[col].median()
                findings_list.append(f"  Avg {col}: {mean_val:.2f} (Median: {median_val:.2f})")
    else:
        findings_list.append(f"No cases found above 90th percentile error threshold for {model_name}.")

def generate_recommendations(findings_list, rf_mae, xgb_mae, current_rule_based_mae=99.73):
    print("\n\n--- Recommendations for run.sh Improvement (from ML Analysis) ---")
    
    best_ml_mae = min(rf_mae, xgb_mae) if rf_mae is not None and xgb_mae is not None else (rf_mae or xgb_mae or float('inf'))
    best_ml_model = "Random Forest" if rf_mae is not None and (xgb_mae is None or rf_mae <= xgb_mae) else "XGBoost"

    recommendations = []
    recommendations.append(f"Current rule-based MAE (for reference): ${current_rule_based_mae:.2f}")
    if best_ml_mae != float('inf'):
        recommendations.append(f"Best ML model ({best_ml_model}) MAE on test set: ${best_ml_mae:.2f}")
        if best_ml_mae < current_rule_based_mae:
            improvement = current_rule_based_mae - best_ml_mae
            recommendations.append(f"**Significant Improvement Potential**: {best_ml_model} model is ~${improvement:.2f} better in MAE (approx {improvement/current_rule_based_mae*100:.1f}% improvement).")
        else:
            recommendations.append(f"ML models did not significantly outperform the current rule-based system on MAE. Focus on insights for rule refinement.")
    else:
        recommendations.append("ML models did not train successfully or MAE was not calculated.")

    recommendations.append("\n**Key Insights & Potential Rule Changes (based on ML model feature importances and tree rules):**")
    
    top_features_from_log = []
    for f_item in findings_list: # Iterate through the entire findings log
        if isinstance(f_item, str):
            if "Top 15 Feature Importances:" in f_item: # Signal start of feature list
                # Reset for each model's list
                current_model_features = [] 
            elif "  - " in f_item and ":" in f_item and "MAE" not in f_item and "Case Index" not in f_item and "Avg " not in f_item:
                try:
                    feature_name = f_item.split("  - ")[1].split(":")[0].strip()
                    if feature_name not in top_features_from_log: # Keep unique features across both models
                        top_features_from_log.append(feature_name)
                except IndexError:
                    pass 
    
    recommendations.append("1. **Refine Segmentation based on Top Features & Decision Tree Rules:**")
    if top_features_from_log:
        recommendations.append(f"   - **Top Influential Features (combined from RF & XGB):** {', '.join(top_features_from_log[:7])} (and their squared/interaction versions). These are critical for segmentation.")
    else:
        recommendations.append(f"   - Examine feature importances (printed above) for critical split points.")
    
    recommendations.append(f"   - **Decision Tree Insights:** Review the printed shallow decision tree rules. They provide explicit split points. For example, if a rule is `|--- total_receipts_amount <= 250.50`, then `250.50` is a candidate for a new boundary in `run.sh`.")
    recommendations.append(f"   - **Multi-Dimensional Segmentation:** The current `run.sh` segments only on `total_receipts_amount`. The ML models indicate that `trip_duration_days`, `miles_traveled`, and derived features like `miles_per_day` (and their interactions/polynomials) are also very important. Consider a nested segmentation strategy:")
    recommendations.append(f"     e.g., Primary split on `total_receipts_amount` (e.g., <=200, 201-1000, >1000).")
    recommendations.append(f"     Then, secondary splits within each on `trip_duration_days` (e.g., 1 day, 2-3 days, 4-6 days, 7+ days).")
    recommendations.append(f"     Then, potentially tertiary splits on `miles_traveled` or `miles_per_day` if tree rules suggest this.")
    recommendations.append(f"     For each final sub-segment, derive a new linear formula (intercept + A*receipts + B*days + C*miles) by fitting a model to just the data points in that sub-segment.")

    recommendations.append("\n2. **Incorporate Non-linearities and Interactions:**")
    recommendations.append(f"   - **Squared Terms:** The importance of features like `total_receipts_amount_sq` means the impact of receipts (and other key variables) is not linear. Within each segment in `run.sh`, consider formulas like: `Base + Coeff1*X + Coeff2*X^2`.")
    recommendations.append(f"   - **Interaction Terms:** If `days_x_receipts` is important, it means the per-receipt reimbursement might change with trip duration. This implies different receipt coefficients for different duration segments.")

    recommendations.append("\n3. **Handle Specific Cases / Outliers Indicated by Error Analysis:**")
    recommendations.append(f"   - Review the 'Error Analysis' sections for both RF and XGB. If certain input combinations (e.g., very long trips with extremely low receipts, or 1-day trips with massive mileage) are consistently mispredicted even by complex models, they might represent unique edge cases in the legacy system that require very specific hardcoded rules or adjustments in `run.sh`.")

    recommendations.append("\n4. **Refine Logic for `total_receipts_amount > 1000`:**")
    recommendations.append(f"   - The current `run.sh` uses a 0.00 coefficient for receipts > $1000. While this was an improvement, the ML models might suggest a more nuanced approach (e.g., a small positive or even a slight negative coefficient for this range, or perhaps a cap on the *total contribution* from receipts rather than just zeroing out the coefficient). Check the decision tree rules for splits around high receipt values.")

    recommendations.append("\n5. **Special Handling for `is_1_day_trip` and `is_5_day_trip`:**")
    recommendations.append(f"   - If these binary flags (or related `duration_cat_` features) show high importance, it confirms that 1-day and 5-day trips have distinct calculation logic. The `run.sh` should have specific base amounts or formula adjustments for these exact durations, potentially overriding the general segmented formulas.")

    recommendations.append("\n**Proposed Action Plan for `run.sh`:**")
    recommendations.append("   a. **Extract Key Split Points:** From the shallow decision tree rules, identify the most common and impactful split points for `total_receipts_amount`, `trip_duration_days`, and `miles_traveled` (or `miles_per_day`).")
    recommendations.append("   b. **Design New Segmentation:** Create 3-5 primary segments in `run.sh` based on these key splits. Start with `total_receipts_amount` as the primary, then `trip_duration_days`.")
    recommendations.append("   c. **Derive Segment-Specific Formulas:** For each new segment, use the `segmentation_analysis.py` script's logic (or manually analyze the data subset) to fit a new linear formula ( `intercept + c1*R + c2*D + c3*M` ). If squared terms were important for a variable in a segment, include it (e.g., `c4*R^2`).")
    recommendations.append("   d. **Test and Iterate:** Implement these new segments and formulas in `run.sh` and use `./eval.sh` to check for improvement. Focus on reducing the MAE.")

    for rec in recommendations:
        print(rec)
        # Add to overall findings_list for record keeping if this function is called within a larger main loop
        if not any(rec_item == rec for rec_item in findings_list): # Avoid duplicates if called multiple times
             findings_list.append(rec)


def main():
    findings = []
    
    df_raw = load_data()
    if df_raw.empty:
        print("Could not load data. Exiting.")
        return
        
    df_featured = advanced_feature_engineering(df_raw)

    excluded_cols = ['expected_output'] 
    # Ensure all features used for training are present in df_featured and are numeric or boolean
    features = [
        col for col in df_featured.columns 
        if col not in excluded_cols and \
           (pd.api.types.is_numeric_dtype(df_featured[col]) or \
            pd.api.types.is_bool_dtype(df_featured[col]))
    ]
    
    X = df_featured[features]
    y = df_featured['expected_output']

    if X.empty or len(X.columns) == 0:
        print("No features available for training after filtering non-numeric or excluded columns. Check feature engineering steps.")
        return
    if len(X) != len(y):
        print(f"Mismatch in length of X ({len(X)}) and y ({len(y)}). Check data processing.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Number of features used for training: {len(features)}")

    rf_model, rf_mae = train_random_forest(X_train, y_train, X_test, y_test, features, findings)
    xgb_model, xgb_mae = train_xgboost(X_train, y_train, X_test, y_test, features, findings)
    
    # For error analysis, use the original df_raw and add key derived features for context
    df_original_inputs_with_derived = df_raw.loc[X.index].copy() # Align with X's index before split
    for col in ['miles_per_day', 'receipt_amount_per_day']: # Add some key derived features for context
        if col in df_featured.columns:
            df_original_inputs_with_derived[col] = df_featured.loc[X.index, col]

    analyze_model_errors(rf_model, X, y, df_original_inputs_with_derived, "Random Forest (on full data)", findings)
    analyze_model_errors(xgb_model, X, y, df_original_inputs_with_derived, "XGBoost (on full data)", findings)

    generate_recommendations(findings, rf_mae, xgb_mae)

    print("\n\n--- Full ML Optimization Findings Log (abbreviated for long rule sets) ---")
    for i, finding_item in enumerate(findings):
        if isinstance(finding_item, str) and "--- Tree" in finding_item and "Rules ---" in finding_item and "\n" in finding_item and len(finding_item) > 500:
            print(f"{i+1}. {finding_item.splitlines()[0]} ... (full ruleset in detailed log if saved, or above)")
        elif isinstance(finding_item, str) and len(finding_item) > 300 and "Top " not in finding_item and "MAE" not in finding_item : 
             print(f"{i+1}. {finding_item[:200]} ... (truncated)")
        else:
            print(f"{i+1}. {finding_item}")
            
    print("\n--- ML Optimization Script Finished ---")

if __name__ == '__main__':
    main()
