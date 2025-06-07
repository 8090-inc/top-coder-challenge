import json
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load data
def load_data(file_path="public_cases.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame([
        {
            'trip_duration_days': item['input']['trip_duration_days'],
            'miles_traveled': item['input']['miles_traveled'],
            'total_receipts_amount': item['input']['total_receipts_amount'],
            'expected_output': item['expected_output']
        } for item in data
    ])
    return df

# --- Baseline Model ---
df = load_data()
X_baseline = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
y = df['expected_output']

baseline_model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

baseline_rmse_scores = []
baseline_r2_scores = []

for train_index, test_index in kf.split(X_baseline):
    X_train, X_test = X_baseline.iloc[train_index], X_baseline.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    baseline_model.fit(X_train, y_train)
    y_pred = baseline_model.predict(X_test)

    baseline_rmse_scores.append(rmse(y_test, y_pred))
    baseline_r2_scores.append(r2_score(y_test, y_pred))

mean_baseline_rmse = np.mean(baseline_rmse_scores)
mean_baseline_r2 = np.mean(baseline_r2_scores)

print("--- Baseline Model Performance ---")
print(f"Mean RMSE: {mean_baseline_rmse:.4f}")
print(f"Mean R2 Score: {mean_baseline_r2:.4f}")

# --- Feature Engineering ---
# Load hypotheses for ideas (optional, as features are specified)
# with open("docs/hypotheses.yaml", 'r') as f:
#     hypotheses = yaml.safe_load(f)

df_eng = df.copy()

# Handle potential division by zero if trip_duration_days can be 0
df_eng['miles_per_day'] = df_eng.apply(lambda row: row['miles_traveled'] / row['trip_duration_days'] if row['trip_duration_days'] > 0 else 0, axis=1)
df_eng['receipt_amount_per_day'] = df_eng.apply(lambda row: row['total_receipts_amount'] / row['trip_duration_days'] if row['trip_duration_days'] > 0 else 0, axis=1)

df_eng['is_5_day_trip'] = (df_eng['trip_duration_days'] == 5).astype(int)

# Handle floating point precision for cents checking
df_eng['ends_in_49_cents'] = df_eng['total_receipts_amount'].apply(lambda x: abs(x - round(x) - 0.49) < 1e-5 or abs(x - (np.floor(x) + 0.49)) < 1e-5 ).astype(int)
df_eng['ends_in_99_cents'] = df_eng['total_receipts_amount'].apply(lambda x: abs(x - round(x) - 0.99) < 1e-5 or abs(x - (np.floor(x) + 0.99)) < 1e-5 ).astype(int)


df_eng['is_short_trip_low_receipts'] = ((df_eng['trip_duration_days'] <= 2) & (df_eng['total_receipts_amount'] < 50)).astype(int)
df_eng['is_long_trip'] = (df_eng['trip_duration_days'] >= 8).astype(int)

# Mileage tiers
bins = [0, 100, 500, np.inf]
labels = ['tier1', 'tier2', 'tier3']
df_eng['mileage_tier'] = pd.cut(df_eng['miles_traveled'], bins=bins, labels=labels, right=True)

# Inspired by H12 (Lisa): Modest expenses on long trips seem to do better on mileage.
df_eng['long_trip_modest_receipts_per_day'] = ((df_eng['is_long_trip'] == 1) & (df_eng['receipt_amount_per_day'] < 100)).astype(int) # Threshold of $100/day

# Inspired by H23 (Kevin): Efficiency sweet spot for miles_per_day (180-220)
df_eng['efficiency_sweet_spot'] = ((df_eng['miles_per_day'] >= 180) & (df_eng['miles_per_day'] <= 220)).astype(int)

# Inspired by H13 (Lisa): Penalized for very low receipts on multi-day trips
df_eng['penalized_low_receipts_multi_day'] = ((df_eng['trip_duration_days'] > 1) & (df_eng['total_receipts_amount'] < 30)).astype(int) # Threshold $30 based on Lisa's comment.


engineered_features_list = [
    'miles_per_day',
    'receipt_amount_per_day',
    'is_5_day_trip',
    'ends_in_49_cents',
    'ends_in_99_cents',
    'is_short_trip_low_receipts',
    'is_long_trip',
    'mileage_tier', # This will be one-hot encoded
    'long_trip_modest_receipts_per_day',
    'efficiency_sweet_spot',
    'penalized_low_receipts_multi_day'
]
print("\n--- Engineered Features Created ---")
for feature in engineered_features_list:
    if feature == 'mileage_tier':
        print(f"- {feature} (to be one-hot encoded: {df_eng['mileage_tier'].unique().tolist()})")
    else:
        print(f"- {feature}")


# --- Enhanced Model ---
# Define categorical and numerical features for preprocessing
categorical_features = ['mileage_tier']
numerical_features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount',
                      'miles_per_day', 'receipt_amount_per_day', 'is_5_day_trip',
                      'ends_in_49_cents', 'ends_in_99_cents', 'is_short_trip_low_receipts',
                      'is_long_trip', 'long_trip_modest_receipts_per_day',
                      'efficiency_sweet_spot', 'penalized_low_receipts_multi_day']

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('passthrough', 'passthrough', numerical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Create the pipeline with preprocessing and the model
enhanced_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_enhanced = df_eng.drop('expected_output', axis=1)

enhanced_rmse_scores = []
enhanced_r2_scores = []

for train_index, test_index in kf.split(X_enhanced):
    X_train, X_test = X_enhanced.iloc[train_index], X_enhanced.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    enhanced_model_pipeline.fit(X_train, y_train)
    y_pred = enhanced_model_pipeline.predict(X_test)

    enhanced_rmse_scores.append(rmse(y_test, y_pred))
    enhanced_r2_scores.append(r2_score(y_test, y_pred))

mean_enhanced_rmse = np.mean(enhanced_rmse_scores)
mean_enhanced_r2 = np.mean(enhanced_r2_scores)

print("\n--- Enhanced Model Performance (with Engineered Features) ---")
print(f"Mean RMSE: {mean_enhanced_rmse:.4f}")
print(f"Mean R2 Score: {mean_enhanced_r2:.4f}")

# Conclusion
print("\n--- Conclusion ---")
if mean_enhanced_rmse < mean_baseline_rmse and mean_enhanced_r2 > mean_baseline_r2:
    print("Engineered features improved the model performance.")
    print(f"RMSE improved from {mean_baseline_rmse:.4f} to {mean_enhanced_rmse:.4f}.")
    print(f"R2 score improved from {mean_baseline_r2:.4f} to {mean_enhanced_r2:.4f}.")
elif mean_enhanced_rmse == mean_baseline_rmse and mean_enhanced_r2 == mean_baseline_r2:
    print("Engineered features did not change the model performance.")
else:
    print("Engineered features did NOT improve the model performance (or performance change was mixed).")
    print(f"Baseline RMSE: {mean_baseline_rmse:.4f}, Enhanced RMSE: {mean_enhanced_rmse:.4f}")
    print(f"Baseline R2: {mean_baseline_r2:.4f}, Enhanced R2: {mean_enhanced_r2:.4f}")
