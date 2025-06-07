# Legacy Reimbursement System Reverse Engineering

This project aims to reverse-engineer ACME Corp's legacy travel reimbursement system by analyzing historical data and employee interviews.

## Project Structure

```
.
├── analysis/              # Analysis code and notebooks
│   ├── scripts/          # Python analysis scripts
│   │   ├── explore_reimbursement_logic.py    # Initial exploration
│   │   └── analyze_initial_findings.py       # Hypothesis comparison
│   └── notebooks/        # Jupyter notebooks for interactive analysis
│       └── explore_reimbursement.ipynb
│
├── data/                 # Data files
│   ├── raw/             # Original data files
│   │   ├── public_cases.json   # 1,000 cases with known outputs
│   │   └── private_cases.json  # 5,000 cases for final testing
│   └── processed/       # Cleaned/transformed data
│
├── results/             # Analysis outputs
│   ├── figures/         # Visualizations and plots
│   │   └── reimbursement_analysis.png
│   ├── reports/         # Analysis reports and findings
│   │   ├── potential_bugs_or_special_cases.csv
│   │   └── analysis_summary.txt
│   └── models/          # Saved models and formulas
│
├── docs/                # Documentation
│   ├── PRD.md          # Product Requirements Document
│   ├── INTERVIEWS.md   # Employee interview transcripts
│   └── hypothesis.txt  # Working hypotheses (living document)
│
├── tests/              # Test scripts for the final model
└── requirements.txt    # Python dependencies
```

## Current Status

### Key Findings
1. **Receipts are the dominant factor** (0.70 correlation), not miles as employees believed
2. **No clear 5-day bonus** despite strong interview claims
3. **Large base intercept** ($915) suggests minimum reimbursement
4. **Low receipt penalty** confirmed (<$50 receipts severely penalized)

### Next Steps
1. Deep dive on receipt processing rules
2. Investigate base amount/minimum
3. Analyze outliers for special rules
4. Test for multiple calculation paths

## Running the Analysis

```bash
# Activate virtual environment
source .venv/bin/activate

# Run initial exploration
python analysis/scripts/explore_reimbursement_logic.py

# Compare findings to hypotheses
python analysis/scripts/analyze_initial_findings.py
```

## Key Documents
- `docs/hypothesis.txt` - Living document tracking all hypotheses and test results
- `docs/INTERVIEWS.md` - Employee insights about the system
- `docs/PRD.md` - Business requirements and success criteria

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

**Good luck and Bon Voyage!**
