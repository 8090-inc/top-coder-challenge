# Legacy Reimbursement Calculator - Reverse Engineering Project

## Project Overview
This project reverse-engineers a legacy travel reimbursement calculation system. Through analysis of 1,000 public test cases, we discovered 6 distinct calculation paths (clusters) that determine reimbursement amounts based on trip days, miles traveled, and receipt totals.

## Key Findings
- **6 Calculation Paths**: Each cluster represents a different type of business trip
- **Special Profile**: 7 cases with specific criteria (7-8 days, 900-1200 miles, 1000-1200 receipts) that follow a unique calculation
- **Current Best Model**: v0.5 achieves MAE of $160.05 (improvement from v0.4's $167.40)

## Directory Structure

```
.
├── calculate_reimbursement.py    # Main calculator with all model versions
├── config.py                     # Project configuration
│
├── data/
│   ├── raw/                      # Original data files
│   │   ├── public_cases.json
│   │   └── private_cases.json
│   ├── processed/                # Processed datasets
│   │   └── public_cases_with_clusters.csv
│   └── predictions/              # Model predictions
│
├── models/
│   └── v0.5_cluster_based/       # Current best model
│       ├── cluster_model_params.json
│       └── decision_tree_*.pkl
│
├── analysis/                     # Analysis scripts (organized by phase)
│   ├── 01_initial_exploration/   # (archived)
│   ├── 02_clustering_discovery/  # Finding 6 clusters
│   ├── 03_cluster_analysis/      # Deep dive into each cluster
│   ├── 04_model_development/     # Building cluster-specific models
│   └── 05_final_implementation/  # v0.5 implementation
│
├── results/
│   ├── figures/                  # Visualizations
│   └── reports/                  # Analysis reports
│
└── archive/                      # Old explorations and iterations
    ├── initial_explorations/
    └── model_iterations/
```

## Model Versions

- **v0.1**: Baseline linear regression (MAE: $189)
- **v0.2**: Inverted receipt coverage model (MAE: $385)
- **v0.3**: Hybrid approach (MAE: $227)
- **v0.4**: Refined with special case handling (MAE: $167)
- **v0.5**: Cluster-based approach (MAE: $160) ⭐ Current best

## Usage

```bash
# Single calculation (uses v0.5 by default)
python calculate_reimbursement.py 7 1000 1100

# Process JSON file
python calculate_reimbursement.py data/raw/public_cases.json

# Use specific version
python calculate_reimbursement.py --version v0.4 5 300 750.50
```

## The 6 Clusters

1. **Standard Multi-Day** (27.6%): 5-12 day trips, low miles, medium receipts
2. **Single Day High Miles** (5.0%): 1-day trips with high mileage (possibly air travel)
3. **Long Trip High Receipts** (29.4%): 10-12 day extended trips
4. **Short Trip** (17.2%): 3-5 day business trips with high daily expenses
5. **Outlier** (0.1%): Single case with very low receipts
6. **Medium Trip High Miles** (20.7%): Contains the special profile cases

## Next Steps

1. Implement full decision tree logic (currently using simplified rules)
2. Fine-tune cluster-specific models
3. Handle edge cases at cluster boundaries
4. Test on private dataset
