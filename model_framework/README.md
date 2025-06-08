# Simple Model Testing Framework

A lightweight framework for testing improvements to reimbursement models.

## Quick Start

### 1. Test the baseline models:
```bash
cd model_framework
python run_experiment.py
```

### 2. Create your own improvement:
```bash
# Edit create_experiment.py with your improvements
python create_experiment.py
```

## Structure

- **core/** - Base classes and evaluation logic
  - `base_model.py` - Abstract model class
  - `evaluator.py` - Model evaluation and comparison
  
- **experiments/** - Experiment tracking
  - `tracker.py` - Simple CSV-based tracking
  
- **models/** - Model implementations
  - `baseline_models.py` - V5 and V3 wrappers
  
- **results/** - Experiment results
  - `experiment_history.csv` - All experiments
  - Individual experiment JSON files

## Creating a New Model

1. Copy the template from `create_experiment.py`
2. Modify the `predict()` method with your improvements
3. Run the script to test
4. Results are automatically saved

## Example Improvements to Try

- Adjust receipt penalty factors (.49 and .99 endings)
- Add corrections for specific clusters
- Boost/reduce predictions for certain trip patterns
- Handle edge cases differently

## Key Metrics

- **MAE** - Mean Absolute Error (primary metric)
- **RMSE** - Root Mean Square Error
- **Bias** - Average prediction error
- **Better/Worse Cases** - Case-by-case comparison

## Tips

- Start with small, targeted improvements
- Test one change at a time
- Check which cases improved/worsened
- Use the experiment history to track progress 