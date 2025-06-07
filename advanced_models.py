import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

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

# --- Feature Engineering (Identical to build_models.py) ---
df = load_data()
df_eng = df.copy()

df_eng['miles_per_day'] = df_eng.apply(lambda row: row['miles_traveled'] / row['trip_duration_days'] if row['trip_duration_days'] > 0 else 0, axis=1)
df_eng['receipt_amount_per_day'] = df_eng.apply(lambda row: row['total_receipts_amount'] / row['trip_duration_days'] if row['trip_duration_days'] > 0 else 0, axis=1)
df_eng['is_5_day_trip'] = (df_eng['trip_duration_days'] == 5).astype(int)
df_eng['ends_in_49_cents'] = df_eng['total_receipts_amount'].apply(lambda x: abs(x - round(x) - 0.49) < 1e-5 or abs(x - (np.floor(x) + 0.49)) < 1e-5 ).astype(int)
df_eng['ends_in_99_cents'] = df_eng['total_receipts_amount'].apply(lambda x: abs(x - round(x) - 0.99) < 1e-5 or abs(x - (np.floor(x) + 0.99)) < 1e-5 ).astype(int)
df_eng['is_short_trip_low_receipts'] = ((df_eng['trip_duration_days'] <= 2) & (df_eng['total_receipts_amount'] < 50)).astype(int)
df_eng['is_long_trip'] = (df_eng['trip_duration_days'] >= 8).astype(int)

bins = [0, 100, 500, np.inf]
labels = ['tier1', 'tier2', 'tier3']
df_eng['mileage_tier'] = pd.cut(df_eng['miles_traveled'], bins=bins, labels=labels, right=True)

df_eng['long_trip_modest_receipts_per_day'] = ((df_eng['is_long_trip'] == 1) & (df_eng['receipt_amount_per_day'] < 100)).astype(int)
df_eng['efficiency_sweet_spot'] = ((df_eng['miles_per_day'] >= 180) & (df_eng['miles_per_day'] <= 220)).astype(int)
df_eng['penalized_low_receipts_multi_day'] = ((df_eng['trip_duration_days'] > 1) & (df_eng['total_receipts_amount'] < 30)).astype(int)

y = df_eng['expected_output']
X_engineered = df_eng.drop('expected_output', axis=1)

# Define categorical and numerical features for preprocessing
# Numerical features will include the original ones and the newly engineered boolean/numerical ones
numerical_features_for_scaling = [
    'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
    'miles_per_day', 'receipt_amount_per_day', 'is_5_day_trip',
    'ends_in_49_cents', 'ends_in_99_cents', 'is_short_trip_low_receipts',
    'is_long_trip', 'long_trip_modest_receipts_per_day',
    'efficiency_sweet_spot', 'penalized_low_receipts_multi_day'
]
categorical_features_for_ohe = ['mileage_tier']

# Create a preprocessor for one-hot encoding categorical features and scaling numerical features
# The order of transformers matters if some generated features are then scaled.
# Here, OHE comes first, then all resulting features (original numerical + OHE + other engineered) are scaled.

# This preprocessor handles OHE for mileage_tier and passes through numericals
# A second step for scaling will be added in the main pipeline for all features.
feature_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_for_ohe)
    ],
    remainder='passthrough' # Keep other columns (numerical ones)
)


# --- Cross-validation setup ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Polynomial Regression Model (Degree 2) ---
# Pipeline: Transform features -> Scale -> PolynomialFeatures -> LinearRegression
poly_model_pipeline = Pipeline(steps=[
    ('feature_transform', feature_transformer), # OHE mileage_tier, pass others
    ('scaler', StandardScaler()),
    ('polyfeatures', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
])

poly_rmse_scores = []
poly_r2_scores = []

for train_index, test_index in kf.split(X_engineered):
    X_train, X_test = X_engineered.iloc[train_index], X_engineered.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    poly_model_pipeline.fit(X_train, y_train)
    y_pred_poly = poly_model_pipeline.predict(X_test)

    poly_rmse_scores.append(rmse(y_test, y_pred_poly))
    poly_r2_scores.append(r2_score(y_test, y_pred_poly))

mean_poly_rmse = np.mean(poly_rmse_scores)
mean_poly_r2 = np.mean(poly_r2_scores)

print("--- Polynomial Regression Model (Degree 2) Performance ---")
print(f"Mean RMSE: {mean_poly_rmse:.4f}")
print(f"Mean R2 Score: {mean_poly_r2:.4f}")


# --- K-Nearest Neighbors (k-NN) Regressor (k=1) ---
# Pipeline: Transform features -> Scale -> KNeighborsRegressor
knn_model_pipeline = Pipeline(steps=[
    ('feature_transform', feature_transformer), # OHE mileage_tier, pass others
    ('scaler', StandardScaler()),
    ('regressor', KNeighborsRegressor(n_neighbors=1, metric='minkowski', p=2)) # p=2 for Euclidean
])

knn_rmse_scores = []
knn_r2_scores = []

for train_index, test_index in kf.split(X_engineered):
    X_train, X_test = X_engineered.iloc[train_index], X_engineered.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    knn_model_pipeline.fit(X_train, y_train)
    y_pred_knn = knn_model_pipeline.predict(X_test)

    knn_rmse_scores.append(rmse(y_test, y_pred_knn))
    knn_r2_scores.append(r2_score(y_test, y_pred_knn))

mean_knn_rmse = np.mean(knn_rmse_scores)
mean_knn_r2 = np.mean(knn_r2_scores)

print("\n--- k-NN Regressor (k=1) Performance ---")
print(f"Mean RMSE: {mean_knn_rmse:.4f}")
print(f"Mean R2 Score: {mean_knn_r2:.4f}")
