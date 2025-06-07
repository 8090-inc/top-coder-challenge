# Baseline and Enhanced Model Performance Report

This report details the performance of a baseline linear regression model and an enhanced linear regression model that incorporates feature engineering for predicting expense reimbursement amounts.

## Data

The models were trained and evaluated on data from `public_cases.json`. The input features are `trip_duration_days`, `miles_traveled`, and `total_receipts_amount`. The target variable is `expected_output`. 5-fold cross-validation was used for evaluation.

## Baseline Model Performance

The baseline model was a simple linear regression using only the raw input features:
*   `trip_duration_days`
*   `miles_traveled`
*   `total_receipts_amount`

**Performance Metrics:**
*   **Mean Root Mean Squared Error (RMSE):** 219.5778
*   **Mean R-squared (R2) Score:** 0.7795

## Feature Engineering

Inspired by hypotheses in `docs/hypotheses.yaml` and the subtask requirements, the following features were engineered:

*   `miles_per_day`: `miles_traveled` / `trip_duration_days`
*   `receipt_amount_per_day`: `total_receipts_amount` / `trip_duration_days`
*   `is_5_day_trip`: Boolean, true if `trip_duration_days` == 5
*   `ends_in_49_cents`: Boolean, true if the cents part of `total_receipts_amount` is .49
*   `ends_in_99_cents`: Boolean, true if the cents part of `total_receipts_amount` is .99
*   `is_short_trip_low_receipts`: Boolean, true if `trip_duration_days` <= 2 and `total_receipts_amount` < 50
*   `is_long_trip`: Boolean, true if `trip_duration_days` >= 8
*   `mileage_tier`: Categorical feature based on `miles_traveled` (0-100: tier1, 101-500: tier2, >500: tier3), subsequently one-hot encoded.
*   `long_trip_modest_receipts_per_day`: Boolean, true if `is_long_trip` is true and `receipt_amount_per_day` < 100. (Inspired by H12)
*   `efficiency_sweet_spot`: Boolean, true if `miles_per_day` is between 180 and 220 (inclusive). (Inspired by H23)
*   `penalized_low_receipts_multi_day`: Boolean, true if `trip_duration_days` > 1 and `total_receipts_amount` < 30. (Inspired by H13)

## Enhanced Model Performance

The enhanced model was a linear regression model using the raw features PLUS all the engineered features listed above.

**Performance Metrics:**
*   **Mean Root Mean Squared Error (RMSE):** 196.9816
*   **Mean R-squared (R2) Score:** 0.8221

## Conclusion

The engineered features **improved the model performance**.
*   The Mean RMSE decreased from 219.5778 (baseline) to 196.9816 (enhanced). A lower RMSE indicates better fit to the data as it measures the average magnitude of the errors.
*   The Mean R2 Score increased from 0.7795 (baseline) to 0.8221 (enhanced). An R2 score closer to 1 indicates that a higher proportion of the variance in the `expected_output` is explained by the model.

These results suggest that the newly created features, based on domain knowledge and hypotheses from employee interviews, provided additional predictive power to the linear regression model.
