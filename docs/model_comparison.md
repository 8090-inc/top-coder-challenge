# Model Comparison Report

This report compares the performance of three different regression models for predicting expense reimbursement amounts: Enhanced Linear Regression, Polynomial Regression (Degree 2), and K-Nearest Neighbors (k-NN, k=1). All models were trained on the `public_cases.json` dataset and utilized the same set of engineered features. Evaluation was performed using 5-fold cross-validation, and features were scaled using `StandardScaler` for Polynomial Regression and k-NN.

## Engineered Features Used

All models listed below used the original raw features (`trip_duration_days`, `miles_traveled`, `total_receipts_amount`) plus the following engineered features:
*   `miles_per_day`
*   `receipt_amount_per_day`
*   `is_5_day_trip`
*   `ends_in_49_cents`
*   `ends_in_99_cents`
*   `is_short_trip_low_receipts`
*   `is_long_trip`
*   `mileage_tier` (one-hot encoded)
*   `long_trip_modest_receipts_per_day`
*   `efficiency_sweet_spot`
*   `penalized_low_receipts_multi_day`

## Performance Metrics

| Model                               | Mean RMSE | Mean R2 Score |
| :---------------------------------- | :-------- | :------------ |
| Enhanced Linear Regression          | 196.9816  | 0.8221        |
| Polynomial Regression (Degree 2)    | 118.7932  | 0.9353        |
| k-NN Regressor (k=1)                | 138.2519  | 0.9125        |

*Enhanced Linear Regression model results are taken from `docs/baseline_report.md`.*

## Conclusion

Based on the performance metrics from 5-fold cross-validation on `public_cases.json`:

*   **Polynomial Regression (Degree 2)** performed the best among the three models. It achieved the lowest Mean RMSE (118.7932) and the highest Mean R2 Score (0.9353). This suggests that allowing for non-linear relationships and interaction terms through polynomial features significantly improved the model's ability to predict reimbursement amounts compared to the linear models.

*   **K-Nearest Neighbors (k-NN, k=1)** was the second-best performing model, with a Mean RMSE of 138.2519 and a Mean R2 Score of 0.9125. This indicates that a non-parametric approach, finding the single most similar past case, can also be very effective for this dataset, outperforming the enhanced linear model.

*   **Enhanced Linear Regression**, while an improvement over the baseline, was outperformed by both Polynomial Regression and k-NN.

Therefore, for the `public_cases.json` dataset, **Polynomial Regression (Degree 2) demonstrates the strongest predictive performance.** The substantial improvement in R2 score (from 0.8221 to 0.9353) suggests that the underlying relationships in the data are better captured by this more complex model. Further tuning (e.g., polynomial degree, k for k-NN) could potentially yield even better results for these models.
