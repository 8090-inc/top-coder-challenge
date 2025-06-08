# Repository Cleanup Summary

## Overview
The repository has been cleaned up to contain only the v5 practical ensemble model and essential files needed for the TopCoder challenge. All other experimental models, analysis, and intermediate files have been moved to the `archive/` directory.

## V5 Model Performance
- **MAE: $77.41** (exceeds target of $110 by 30%)
- **Approach**: Practical ensemble combining rule engine (v3) with ML residual correction
- **Key Features**:
  - Conservative ML corrections to avoid overfitting
  - Uses ExtraTrees and GradientBoosting models
  - Strong regularization (max depth 8, min samples 20)
  - Maximum correction capped at $100

## Current Directory Structure

```
top-coder-challenge/
├── models/                          # V5 model and dependencies only
│   ├── v5_practical_ensemble.py     # Main v5 model implementation
│   ├── v5_practical_ensemble.pkl    # Trained model file
│   ├── train_v5_model.py           # Script to retrain v5
│   ├── cluster_models_optimized.py  # V3 rule engine (dependency)
│   ├── cluster_router.py           # Cluster assignment logic
│   └── __init__.py
├── archive/                        # All archived content
│   ├── models/                     # All other model versions
│   ├── analysis/                   # Analysis scripts and visualizations
│   ├── predictions/                # Old prediction files
│   ├── test_scripts/               # Test and validation scripts
│   └── ...
├── calculate_reimbursement.py      # Main entry point (uses v5)
├── public_cases.json              # Original test data
├── private_cases.json             # Original private data
├── public_cases_expected_outputs.csv # Expected outputs
├── eval.sh                        # Evaluation script
├── generate_results.sh            # Results generation script
├── run.sh                         # Run script
├── requirements.txt               # Python dependencies
└── README.md                      # Original README

```

## Usage

### Single Prediction
```bash
python calculate_reimbursement.py <trip_days> <miles> <receipts>
# Example: python calculate_reimbursement.py 5 300 750.50
```

### Batch Processing
```bash
python calculate_reimbursement.py public_cases.json
```

### Generating Challenge Submission
```bash
./generate_results.sh
```

## Key Files Retained

1. **V5 Model Files**:
   - `models/v5_practical_ensemble.py` - Main model implementation
   - `models/v5_practical_ensemble.pkl` - Trained model (1.2MB)
   - `models/train_v5_model.py` - Training script if retraining needed

2. **Dependencies**:
   - `models/cluster_models_optimized.py` - V3 rule engine used by v5
   - `models/cluster_router.py` - Cluster assignment logic

3. **Challenge Files**:
   - All original JSON data files
   - Evaluation and generation scripts
   - Configuration files

## Archived Content

Everything else has been moved to `archive/` including:
- V6 models and attempts (showed regression from v5)
- All analysis scripts and visualizations
- Test scripts and error analysis
- Documentation and reports
- Intermediate model versions

## Notes

- The v5 model automatically loads when `calculate_reimbursement.py` is run
- If the model file is missing, retrain using `python models/train_v5_model.py`
- The model uses all 1000 public cases for training (no holdout)
- Conservative approach prevents overfitting while achieving strong performance 