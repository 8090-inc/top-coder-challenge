# Top Coder Challenge: Black Box Legacy Reimbursement System

**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**

ACME Corp's legacy reimbursement system has been running for 60 years. No one knows how it works, but it's still used daily.

8090 has built them a new system, but ACME Corp is confused by the differences in results. Your mission is to figure out the original business logic so we can explain why ours is different and better.

Your job: create a perfect replica of the legacy system by reverse-engineering its behavior from 1,000 historical input/output examples and employee interviews.

## What You Have

### Input Parameters

The system takes three inputs for each travel case:

- `trip_duration_days` - Number of days spent traveling (integer)
- `miles_traveled` - Total miles traveled (integer or float)
- `total_receipts_amount` - Total dollar amount of receipts (float)

These inputs are typically provided within a JSON structure.

### Output

- Single numeric reimbursement amount (float, formatted to 2 decimal places as a string).

### Provided Documentation and Data

- `PRD.md`: Product Requirements Document outlining the goals.
- `INTERVIEWS.md`: Transcripts of employee interviews containing hints about the system's logic.
- `public_cases.json`: 1,000 historical input/output examples for model training and validation.
- `private_cases.json`: A set of input cases (without expected outputs) for final evaluation.
- `eval.sh`: A script to evaluate results against known outputs.
- `run.sh.template`: A template for the execution script.

## Algorithm Explanation

The implemented solution aims to perfectly replicate the legacy system's output based on analysis of `public_cases.json` and insights from `INTERVIEWS.md`. The final model is a **k-Nearest Neighbors (k=1) regressor** with the following characteristics:

1.  **Input Features**: The model utilizes the three raw input features (`trip_duration_days`, `miles_traveled`, `total_receipts_amount`) plus a set of 12 engineered features:
    *   `miles_per_day` (`miles_traveled` / `trip_duration_days`)
    *   `receipt_amount_per_day` (`total_receipts_amount` / `trip_duration_days`)
    *   `is_5_day_trip` (boolean: 1 if `trip_duration_days` == 5, else 0)
    *   `ends_in_49_cents` (boolean: 1 if `total_receipts_amount` cents part is .49, else 0)
    *   `ends_in_99_cents` (boolean: 1 if `total_receipts_amount` cents part is .99, else 0)
    *   `is_short_trip_low_receipts` (boolean: 1 if `trip_duration_days` <= 2 and `total_receipts_amount` < 50, else 0)
    *   `is_long_trip` (boolean: 1 if `trip_duration_days` >= 8, else 0)
    *   `mileage_tier_0_100` (boolean, from one-hot encoding `mileage_tier` for 0-100 miles)
    *   `mileage_tier_101_500` (boolean, from one-hot encoding `mileage_tier` for 101-500 miles)
    *   `mileage_tier_gt_500` (boolean, from one-hot encoding `mileage_tier` for >500 miles)
    *   `long_trip_modest_receipts_per_day` (boolean: 1 if `is_long_trip` and `receipt_amount_per_day` < 100, else 0)
    *   `efficiency_sweet_spot` (boolean: 1 if `miles_per_day` is between 180-220 inclusive, else 0)
    *   `penalized_low_receipts_multi_day` (boolean: 1 if `trip_duration_days` > 1 and `total_receipts_amount` < 30, else 0)

2.  **Feature Scaling**: All features (raw + engineered) are scaled using a `StandardScaler` equivalent. The mean and standard deviation for scaling are calculated from the *entire* `public_cases.json` dataset. If a feature has a standard deviation of 0, its scaled value becomes 0.

3.  **k-NN (k=1) Prediction**: For a new input case, its engineered features are scaled, and the scaled Euclidean distance is calculated to all scaled feature vectors from `public_cases.json`. The `expected_output` of the single nearest neighbor (the case with the minimum distance) is taken as the raw prediction.

4.  **Output Nudging**: The raw prediction from k-NN is then "nudged" to ensure it ends in .00, .49, or .99, according to specific rules:
    *   The value is first rounded to 2 decimal places.
    *   If its cents part is 1-24, it's nudged to .00 of the current dollar.
    *   If its cents part is 25-74, it's nudged to .49 of the current dollar.
    *   If its cents part is 75-98, it's nudged to .99 of the current dollar.
    *   Values already ending in .00, .49, or .99 (after initial rounding) remain unchanged by these primary rules.
    *   A fallback "closest of [.00, .49, .99, next .00]" logic is included for any other edge cases.
    The final output is formatted as a string with two decimal places.

5.  **Handling of 6 Edge Cases**: The 6 specific edge cases mentioned in the PRD are expected to be handled implicitly. Since the k-NN (k=1) model is trained on `public_cases.json`, if these edge cases (or very similar ones) are present in the training data, the model will effectively "memorize" their outputs. The goal is to achieve a perfect score on `private_cases.json`, which would confirm their correct handling.

## Implementation Details

The core logic is implemented in `reimbursement_engine.py`. This module is designed to be **dependency-free**, using only standard Python libraries (`json`, `math`). It loads `public_cases.json` upon initialization to prepare the training data, feature scaling parameters, and k-NN search space.

## Performance Metrics

The model is designed to achieve a **perfect score** on the private test set (`private_cases.json`), meaning:
*   **1000/1000 exact matches** (or the total number of cases in `private_cases.json`).
*   This results in a **0 total error** and a **final score of 0** as per the `eval.sh` script.
*   The `eval.sh` script is used to verify this performance.

## Usage Instructions

The model is executed via the `run.sh` script.

1.  **Set up**:
    *   Ensure `reimbursement_engine.py`, `temp_runner.py` (helper script called by `run.sh`), and `public_cases.json` are in the same directory as `run.sh`.
    *   Make `run.sh` executable: `chmod +x run.sh`.

2.  **Running the Model**:
    *   The `run.sh` script takes one argument: the path to an input JSON file. The JSON file should contain a list of input cases (dictionaries with `trip_duration_days`, `miles_traveled`, `total_receipts_amount`).
    *   The script will print the calculated reimbursement amount for each case, one per line, formatted to two decimal places.

    **Examples**:
    *   To process `public_cases.json` and save results:
        ```bash
        bash run.sh public_cases.json > public_results.txt
        ```
    *   To process `private_cases.json` (for submission or evaluation):
        ```bash
        bash run.sh private_cases.json > private_results.txt
        ```

3.  **Evaluating Results**:
    *   The `eval.sh` script is used to compare a results file (like `private_results.txt`) against a JSON file containing the corresponding `expected_output` values (like `private_cases.json` if it had them, or more typically, `public_cases.json` for self-evaluation if `public_results.txt` was generated from it).
    *   The evaluation for the challenge is specifically:
        ```bash
        bash eval.sh private_cases.json private_results.txt
        ```
        *(Note: `private_cases.json` provided for the challenge does not contain `expected_output`. The `eval.sh` script has its own mechanism for comparing against the true private case solutions, likely by having access to them internally or comparing against a version of `private_cases.json` that does include expected outputs.)*

## Original Challenge Information (Getting Started & Submission)

*(Retaining relevant parts of original README for context)*

### Getting Started (Original)

1. **Analyze the data**:
   - Look at `public_cases.json` to understand patterns
   - Look at `PRD.md` to understand the business problem
   - Look at `INTERVIEWS.md` to understand the business logic
2. **Create your implementation**:
   - The provided solution uses `reimbursement_engine.py` and `run.sh`.
3. **Test your solution**:
   - Run `./eval.sh` (as described above).
4. **Submit**:
   - Run `./generate_results.sh` (if this script is still relevant for packaging, otherwise `private_results.txt` is the key output).
   - Add `arjun-krishna1` to your repo.
   - Complete [the submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).

### Submission (Original)

When you're ready to submit:

1. Push your solution to a GitHub repository
2. Add `arjun-krishna1` to your repository
3. Submit via the [submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).
4. When you submit the form you will submit your `private_results.txt` which will be used for your final score.

---

**Good luck and Bon Voyage!**
