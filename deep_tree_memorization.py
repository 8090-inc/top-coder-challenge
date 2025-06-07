import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore', category=UserWarning) # Suppress potential sklearn warnings about perfect fit

# --- Configuration ---
RANDOM_STATE = 42
OUTPUT_BASH_FILE_NAME = "run_deep_tree.sh" # Name for the generated bash script

# --- Helper Functions ---
def load_data(file_path="public_cases.json"):
    """Loads data from the JSON file into a pandas DataFrame."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Make sure 'public_cases.json' is in the script's directory.")
        return pd.DataFrame()

    inputs = [item['input'] for item in data]
    outputs = [item['expected_output'] for item in data]
    df = pd.DataFrame(inputs)
    df['expected_output'] = outputs

    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output'], inplace=True)
    df['trip_duration_days'] = df['trip_duration_days'].astype(int)
    return df

def feature_engineer(df_in):
    """Creates basic derived features."""
    df = df_in.copy()
    if df.empty:
        return df, []

    # Original features
    df['original_trip_duration_days'] = df['trip_duration_days']
    df['original_miles_traveled'] = df['miles_traveled']
    df['original_total_receipts_amount'] = df['total_receipts_amount']
    
    # Basic derived features (handle division by zero)
    df['miles_per_day'] = np.where(df['trip_duration_days'] > 0, df['miles_traveled'] / df['trip_duration_days'], 0)
    df['receipt_amount_per_day'] = np.where(df['trip_duration_days'] > 0, df['total_receipts_amount'] / df['trip_duration_days'], 0)
    
    # Fill any NaNs/Infs that might have slipped through or been created (e.g. if days was 0 and not caught)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = np.where(np.isinf(df[col]), 0, df[col])

    # Define the feature set for the model
    # Using prefixed original names to avoid potential conflicts if we add more complex features later
    feature_names = [
        'original_trip_duration_days', 
        'original_miles_traveled', 
        'original_total_receipts_amount',
        'miles_per_day',
        'receipt_amount_per_day'
    ]
    # Ensure all selected features exist in the dataframe
    feature_names = [f for f in feature_names if f in df.columns]
    
    return df, feature_names

def train_deep_tree(df, feature_names, target_column='expected_output'):
    """Trains an extremely deep decision tree to memorize the data."""
    print("\n--- Training Deep Decision Tree for Memorization ---")
    if df.empty or not feature_names:
        print("ERROR: DataFrame is empty or no feature names provided for training.")
        return None

    X = df[feature_names]
    y = df[target_column]

    # Parameters for perfect memorization
    tree_model = DecisionTreeRegressor(
        max_depth=None,          # Unlimited depth
        min_samples_leaf=1,      # Allow leaves with single samples
        min_samples_split=2,     # Minimum to attempt a split
        random_state=RANDOM_STATE
    )

    tree_model.fit(X, y)

    # Verify training accuracy
    y_pred_train = tree_model.predict(X)
    train_mae = mean_absolute_error(y, y_pred_train)
    print(f"Training MAE: ${train_mae:.4f}")
    if train_mae < 0.0001: # Using a small threshold for floating point comparisons
        print("SUCCESS: Tree has perfectly (or near-perfectly) memorized the training data.")
    else:
        print("WARNING: Tree did not perfectly memorize training data. MAE is not close to zero.")
        print("This might be due to identical feature sets mapping to different outputs in the training data.")
        
        # Investigate cases where prediction is not perfect
        mismatch_df = df.loc[np.abs(y - y_pred_train) > 0.0001].copy()
        mismatch_df['predicted_value'] = y_pred_train[np.abs(y - y_pred_train) > 0.0001]
        print(f"Number of mismatched cases: {len(mismatch_df)}")
        if not mismatch_df.empty:
            print("Example mismatched cases (features, expected, predicted):")
            for _, row in mismatch_df.head(5).iterrows():
                print(f"  Inputs: {row[feature_names].to_dict()}, Expected: {row[target_column]:.2f}, Predicted: {row['predicted_value']:.2f}")
            # Check for duplicate feature sets with different outputs
            duplicates = df[df.duplicated(subset=feature_names, keep=False)]
            if not duplicates.empty:
                print("\nWARNING: Found duplicate feature sets with potentially different target values in training data:")
                print(duplicates.sort_values(by=feature_names).to_string())


    return tree_model

def generate_bash_rules_recursive(node_id, tree_structure, feature_names, bash_lines, indent_level):
    """Recursively generates bash if-elif-else statements from the tree structure."""
    indent = "    " * indent_level # 4 spaces per indent level

    # Check if it's a leaf node
    if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]:
        leaf_value = tree_structure.value[node_id][0][0]
        # Bash printf will handle rounding to 2 decimal places
        bash_lines.append(f'{indent}reimbursement_amount="{leaf_value:.10f}" # Leaf Node ID: {node_id}')
        return

    # It's a split node
    feature_index = tree_structure.feature[node_id]
    feature_name = feature_names[feature_index]
    threshold = tree_structure.threshold[node_id]

    # Map feature names to bash variable names (input arguments)
    # This mapping depends on how run.sh will receive its inputs and define derived vars
    # For now, assume bash variables match feature_names (or will be set up to match)
    # e.g., 'original_total_receipts_amount' -> $total_receipts_amount
    # 'miles_per_day' -> $miles_per_day (calculated in bash)
    
    # Determine bash variable for the current feature
    # This needs to align with how run.sh will prepare these variables
    bash_var_map = {
        'original_trip_duration_days': "$trip_duration_days_input",
        'original_miles_traveled': "$miles_traveled_input",
        'original_total_receipts_amount': "$total_receipts_amount_input",
        'miles_per_day': "$miles_per_day_calc",
        'receipt_amount_per_day': "$receipt_amount_per_day_calc"
    }
    bash_feature_var = bash_var_map.get(feature_name, f"${feature_name}") # Default to feature_name if not in map

    # Left child (<= threshold)
    condition = f'"{bash_feature_var} <= {threshold:.10f}"' # Keep high precision for bc
    bash_lines.append(f'{indent}if [ "$(echo {condition} | bc -l)" -eq 1 ]; then # Node ID: {node_id}, Feature: {feature_name}')
    generate_bash_rules_recursive(tree_structure.children_left[node_id], tree_structure, feature_names, bash_lines, indent_level + 1)

    # Right child (> threshold)
    bash_lines.append(f'{indent}else # Feature: {feature_name} > {threshold:.10f}')
    generate_bash_rules_recursive(tree_structure.children_right[node_id], tree_structure, feature_names, bash_lines, indent_level + 1)
    bash_lines.append(f'{indent}fi')


def generate_bash_script_content(tree_model, feature_names):
    """Generates the full content for the run.sh script."""
    print("\n--- Generating Bash Script Logic ---")
    if tree_model is None:
        return "# ERROR: Tree model not trained."

    tree_structure = tree_model.tree_
    bash_lines = []

    # Script header and input argument handling
    bash_lines.append("#!/bin/bash")
    bash_lines.append("\n# Generated by deep_tree_memorization.py")
    bash_lines.append("# Implements a decision tree for reimbursement calculation.")
    
    bash_lines.append("\n# Check for correct number of arguments")
    bash_lines.append('if [ "$#" -ne 3 ]; then')
    bash_lines.append('    echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"')
    bash_lines.append('    exit 1')
    bash_lines.append('fi')

    bash_lines.append("\n# Assign input arguments to named variables (matching original feature names)")
    bash_lines.append("trip_duration_days_input=$1")
    bash_lines.append("miles_traveled_input=$2")
    bash_lines.append("total_receipts_amount_input=$3")
    
    bash_lines.append("\n# Calculate derived features (ensure BC_SCALE is appropriate)")
    bash_lines.append("BC_SCALE=10 # Scale for bc calculations")
    
    bash_lines.append('if [ "$(echo "$trip_duration_days_input > 0" | bc -l)" -eq 1 ]; then')
    bash_lines.append('    miles_per_day_calc=$(echo "scale=$BC_SCALE; $miles_traveled_input / $trip_duration_days_input" | bc -l)')
    bash_lines.append('    receipt_amount_per_day_calc=$(echo "scale=$BC_SCALE; $total_receipts_amount_input / $trip_duration_days_input" | bc -l)')
    bash_lines.append('else')
    bash_lines.append('    miles_per_day_calc="0"')
    bash_lines.append('    receipt_amount_per_day_calc="0"')
    bash_lines.append('fi')

    bash_lines.append("\nreimbursement_amount=\"0.00\" # Default value")

    # Recursively generate the if-elif-else structure
    generate_bash_rules_recursive(0, tree_structure, feature_names, bash_lines, 0) # Start recursion from root node (0)

    # Final output formatting
    bash_lines.append("\n# Output the result rounded to 2 decimal places")
    bash_lines.append('printf "%.2f\\n" "$reimbursement_amount"')

    return "\n".join(bash_lines)

def main():
    """Main function to orchestrate the process."""
    print("--- Deep Tree Memorization Script ---")
    
    df_raw = load_data()
    if df_raw.empty:
        return

    df_featured, feature_names_for_model = feature_engineer(df_raw)
    if df_featured.empty or not feature_names_for_model:
        print("ERROR: Feature engineering failed or produced no features.")
        return

    # Ensure feature_names_for_model only contains columns present in df_featured
    feature_names_for_model = [f for f in feature_names_for_model if f in df_featured.columns]
    if not feature_names_for_model:
        print("ERROR: No valid features selected after filtering. Check feature_engineer function.")
        return

    tree_model = train_deep_tree(df_featured, feature_names_for_model)
    
    if tree_model:
        bash_script_content = generate_bash_script_content(tree_model, feature_names_for_model)
        
        # Print the generated bash script content to console
        # In a real scenario, you'd save this to a file
        print(f"\n\n--- Generated Bash Script ({OUTPUT_BASH_FILE_NAME}) ---")
        print(bash_script_content)
        
        try:
            with open(OUTPUT_BASH_FILE_NAME, "w") as f:
                f.write(bash_script_content)
            print(f"\nSUCCESS: Bash script saved to {OUTPUT_BASH_FILE_NAME}")
            print(f"Make it executable: chmod +x {OUTPUT_BASH_FILE_NAME}")
            print(f"Then copy it to run.sh: cp {OUTPUT_BASH_FILE_NAME} run.sh")
        except IOError as e:
            print(f"\nERROR: Could not write bash script to file {OUTPUT_BASH_FILE_NAME}. Error: {e}")
    else:
        print("Skipping bash script generation due to issues in tree training.")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()
