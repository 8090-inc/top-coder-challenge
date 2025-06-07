# Exploratory Data Analysis (EDA) of Expense Reimbursement Data

This document summarizes the findings from an exploratory data analysis of the `public_cases.json` dataset. The dataset contains information about business trips, including duration, mileage, and receipt totals, along with the corresponding expected reimbursement amount.

## Summary Statistics

Below are the summary statistics for each input feature and the expected output:

### Trip Duration (days)

-   **Min:** 1.00
-   **Max:** 14.00
-   **Mean:** 7.04
-   **Median:** 7.00
-   **StdDev:** 3.93

### Miles Traveled

-   **Min:** 5.00
-   **Max:** 1317.07
-   **Mean:** 597.41
-   **Median:** 621.00
-   **StdDev:** 351.30

### Total Receipts Amount

-   **Min:** 1.42
-   **Max:** 2503.46
-   **Mean:** 1211.06
-   **Median:** 1171.90
-   **StdDev:** 742.85

### Expected Output (Reimbursement Amount)

-   **Min:** 117.24
-   **Max:** 2337.73
-   **Mean:** 1349.11
-   **Median:** 1454.26
-   **StdDev:** 470.32

## Potential Correlations and Observations

The following correlations were observed between the input features and the `expected_output`:

-   **`total_receipts_amount` vs `expected_output`:** Correlation coefficient of **0.704**. This indicates a strong positive correlation, suggesting that as the total amount on receipts increases, the expected reimbursement generally increases.
-   **`trip_duration_days` vs `expected_output`:** Correlation coefficient of **0.514**. This indicates a moderate positive correlation, suggesting that longer trips tend to result in higher reimbursement amounts.
-   **`miles_traveled` vs `expected_output`:** Correlation coefficient of **0.432**. This indicates a moderate positive correlation, suggesting that trips with higher mileage tend to receive higher reimbursements, though the relationship is not as strong as with receipt amounts or trip duration.

### Brief Observations:

*   **`expected_output` and `trip_duration_days`**: Expected output generally increases with `trip_duration_days`.
*   **`expected_output` and `miles_traveled`**: The correlation between `expected_output` and `miles_traveled` is moderate (0.432). While there's a tendency for reimbursement to increase with mileage, other factors might play a more significant role, or the relationship might be non-linear.
*   **`expected_output` and `total_receipts_amount`**: Expected output generally increases with `total_receipts_amount`. This is the strongest linear relationship observed among the input features.

Further investigation, including visualization and analysis of interaction effects, would be beneficial to uncover more complex patterns and non-linear relationships.
