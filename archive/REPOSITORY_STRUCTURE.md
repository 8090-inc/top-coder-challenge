# Repository Structure

## Overview
This repository contains the solution for the legacy travel reimbursement calculation challenge. After extensive analysis and experimentation, we've achieved MAE $77.41 with the V5 Practical Ensemble model.

## Directory Structure

```
top-coder-challenge/
├── calculate_reimbursement.py      # Main entry point for predictions
├── requirements.txt                # Python dependencies
├── config.py                      # Configuration settings
├── README.md                      # Project overview
├── eval.sh                        # Evaluation script
├── generate_results.sh            # Generate submission results
├── run.sh                         # Run predictions
│
├── models/                        # Production models
│   ├── v5_practical_ensemble.py   # Latest model (MAE: $77.41)
│   ├── v5_practical_ensemble.pkl  # Trained model weights
│   ├── train_v5_model.py         # Training script
│   ├── cluster_models_optimized.py # V3 cluster-specific models
│   └── cluster_router.py         # Cluster assignment logic
│
├── analysis/                      # Active research
│   ├── 09_cents_hash_discovery/  # Working on cents pattern discovery
│   └── 10_per_cluster_ensemble/  # Baseline reference implementation
│
├── data/                         
│   └── raw/                      # Original data files
│       ├── public_cases.json     # 1,000 public test cases
│       └── private_cases.json    # 5,000 private evaluation cases
│
├── docs/                         # Documentation
│   ├── PRD.md                   # Product requirements
│   ├── INTERVIEWS.md            # Employee interview insights
│   └── hypothesis.txt           # Research findings & hypotheses
│
├── results/                      # Analysis outputs
│   ├── figures/                 # Visualization plots
│   └── reports/                 # CSV reports and analysis
│
└── archive/                     # Old experiments and versions
```

## Key Components

### 1. Production Model (V5)
- **File**: `models/v5_practical_ensemble.py`
- **Performance**: MAE $77.41 on public cases
- **Description**: Ensemble combining V3 optimized clusters with residual corrector
- **Usage**: Automatically loaded by `calculate_reimbursement.py`

### 2. Cluster Models (V3)
- **File**: `models/cluster_models_optimized.py`
- **Clusters**: 9 distinct calculation patterns identified
- **Router**: `cluster_router.py` assigns cases to appropriate clusters

### 3. Active Research
- **Cents Hash Discovery**: Working on understanding the precise cents patterns (.12, .24, .72, .94)
- **Location**: `analysis/09_cents_hash_discovery/`

### 4. Running Predictions
```bash
# Generate predictions for private cases
./generate_results.sh

# Or run directly
python calculate_reimbursement.py
```

## Model Evolution Summary
- V0.1: Basic linear model (MAE: $189.90)
- V0.5: Initial cluster-based (MAE: $160.05)
- V2: Extended clusters (MAE: $101.13)
- V3: Optimized clusters (MAE: $77.95)
- V5: Practical ensemble (MAE: $77.41) ← Current production

## Next Steps
1. Complete cents hash discovery to potentially achieve exact matches
2. Consider implementing cleaner model versioning system
3. Further optimize cluster boundaries if new patterns emerge 