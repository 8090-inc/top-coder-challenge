# Top Coder Challenge: Black Box Legacy Reimbursement System

**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**

ACME Corp's legacy reimbursement system has been running for 60 years. No one knows how it works, but it's still used daily.

8090 has built them a new system, but ACME Corp is confused by the differences in results. Your mission is to figure out the original business logic so we can explain why ours is different and better.

Your job: create a perfect replica of the legacy system by reverse-engineering its behavior from 1,000 historical input/output examples and employee interviews.

## What You Have

### Input Parameters

The system takes three inputs:

- `trip_duration_days` - Number of days spent traveling (integer)
- `miles_traveled` - Total miles traveled (integer)
- `total_receipts_amount` - Total dollar amount of receipts (float)

## Documentation

- A PRD (Product Requirements Document)
- Employee interviews with system hints

### Output

- Single numeric reimbursement amount (float, rounded to 2 decimal places)

### Historical Data

- `public_cases.json` - 1,000 historical input/output examples

## Getting Started

1. **Analyze the data**: 
   - Look at `public_cases.json` to understand patterns
   - Look at `PRD.md` to understand the business problem
   - Look at `INTERVIEWS.md` to understand the business logic
2. **Create your implementation**:
   - Copy `run.sh.template` to `run.sh`
   - Implement your calculation logic
   - Make sure it outputs just the reimbursement amount
3. **Test your solution**: 
   - Run `./eval.sh` to see how you're doing
   - Use the feedback to improve your algorithm
4. **Submit**:
   - Run `./generate_results.sh` to get your final results.
   - Add `arjun-krishna1` to your repo.
   - Complete [the submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).

## Implementation Requirements

Your `run.sh` script must:

- Take exactly 3 parameters: `trip_duration_days`, `miles_traveled`, `total_receipts_amount`
- Output a single number (the reimbursement amount)
- Run in under 5 seconds per test case
- Work without external dependencies (no network calls, databases, etc.)

Example:

```bash
./run.sh 5 250 150.75
# Should output something like: 487.25
```

## Evaluation

Run `./eval.sh` to test your solution against all 1,000 cases. The script will show:

- **Exact matches**: Cases within ±$0.01 of the expected output
- **Close matches**: Cases within ±$1.00 of the expected output
- **Average error**: Mean absolute difference from expected outputs
- **Score**: Lower is better (combines accuracy and precision)

Your submission will be tested against `private_cases.json` which does not include the outputs.

## Submission

When you're ready to submit:

1. Push your solution to a GitHub repository
2. Add `arjun-krishna1` to your repository
3. Submit via the [submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).
4. When you submit the form you will submit your `private_results.txt` which will be used for your final score.

---

## Approaches to solve this code challenge

The repository contains a coding challenge to reproduce the behavior of ACME Corp’s legacy travel reimbursement system. The README explains that you must reverse‑engineer the original logic from 1,000 historical cases and employee interviews, ultimately implementing a run.sh script that outputs a single reimbursement amount when given trip duration, miles traveled, and receipt total.

The PRD emphasizes that the legacy system is a “black box” with unpredictable behaviors that must be replicated exactly, even if they stem from quirks or bugs. Employee interviews describe suspected patterns: bonuses for 5‑day trips with around 180–220 miles per day, penalties for long high‑spending trips, and other non‑linear effects.

Observations from the Public Data:
The dataset contains 1,000 cases with trip_duration_days (1–14), miles_traveled (mostly integer, some decimals), and total_receipts_amount (floating point).
Simple correlations show that reimbursement per day is strongly correlated with receipts per day (~0.93) and also correlated with miles per day (~0.80).
Average reimbursement rises with trip length but has dips (e.g., a drop around 8 days), suggesting thresholds or penalties rather than a single linear formula.

Plausible Approaches in Python

1. Rule‑Based Modeling
Analyze the data and interview hints to infer per‑day allowances, mileage bands, and receipt caps.
Implement piecewise logic:
Base reimbursement per day (different rates for 1–4, 5–7, etc.).
Mileage multipliers that drop off after certain thresholds (e.g., after ~200 miles/day).
Receipt reimbursement with diminishing returns and special bonuses/penalties (e.g., the “sweet spot combo” for 5‑day trips under $100/day).
This yields a deterministic script that captures known quirks and is easy to explain.
This approach matches the interviews’ emphasis on specific thresholds and “sweet spots,” and it keeps the implementation lightweight (no external ML dependencies), satisfying the requirement that run.sh should run quickly and without extra packages.

2. Machine Learning Approximation
Treat the inputs as features and train a regression model (e.g., RandomForest or GradientBoosting) using the 1,000 public cases.
Include engineered features such as miles_per_day, receipts_per_day, interaction terms, and polynomial expansions.
Measure error using the provided eval.sh script; iterate until the model predicts within a small average error.
For reproducibility, the trained model’s parameters can be exported to a small Python script (e.g., using sklearn’s model serialization or embedding the decision tree rules directly). This can closely match the legacy outputs without manually deriving every rule.
Both approaches make sense because the interviews strongly suggest complex, non‑linear behavior with bonuses, penalties, and diminishing returns. Rule‑based logic provides transparency and might mimic intentional business rules, while an ML model can capture subtle patterns that are difficult to deduce manually. The best solution may combine both—start with interpretable rules informed by the interviews, then use machine learning to fine‑tune or account for residual patterns.

Either way, the core steps remain:
1. Explore public_cases.json to identify relationships among days, miles, and receipts.
2. Build a Python implementation that computes the reimbursement using those insights.
3. Test against all cases via ./eval.sh and iterate until the score is sufficiently low.

This approach fulfills the challenge of replicating the legacy system’s outputs while adhering to the repository’s guidelines.
