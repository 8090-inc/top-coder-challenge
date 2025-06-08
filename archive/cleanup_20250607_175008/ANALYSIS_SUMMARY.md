# Analysis Summary

## Phase 1: Initial Exploration
- Discovered non-linear relationships
- Found receipt amount inversely affects reimbursement
- Identified special handling for .49/.99 receipt endings

## Phase 2: Clustering Discovery
- Applied K-means clustering with 7 derived features
- Found 6 distinct clusters matching Kevin's observation
- Identified cluster 5 contains special profile

## Phase 3: Cluster Analysis
- Deep dive into each cluster's characteristics
- Found special profile uses step function based on receipts
- Confirmed each cluster represents different trip type

## Phase 4: Model Development
- Tested linear regression, decision trees for each cluster
- Found decision trees work best for most clusters
- Achieved theoretical MAE of $101.36

## Phase 5: Final Implementation
- Implemented v0.5 in calculate_reimbursement.py
- Simplified decision tree rules for portability
- Current performance: MAE $160.05

## Key Files to Review
1. `analysis/05_final_implementation/final_cluster_summary.py` - Best overview
2. `models/v0.5_cluster_based/cluster_model_params.json` - Model parameters
3. `results/reports/final_cluster_summary.csv` - Cluster statistics
