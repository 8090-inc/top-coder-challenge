# V5 Model Improvement Strategy

## Current Performance
- **MAE**: $77.41 (already exceeds $110 target)
- **MAPE**: 6.8%
- **Max Error**: $362.18

## Top 3 Improvements to Implement

### 1. Fix Receipt Penalty Handling (~5-8% improvement)
- .99 endings have 85% higher error rate
- Ensure v5 correctly applies 0.341x (.49) and 0.51x (.99) multipliers
- This affects only 29 cases but with high impact

### 2. Add Cents Classification (~3-5% improvement)
- 988/1000 outputs have precise cents patterns
- Train ExtraTreesClassifier on cents prediction
- Current v5 gets 0 exact matches

### 3. Target Specific Outliers (Not Blanket Corrections)
Instead of broad pattern corrections, target specific cases:
- **Case 86 pattern**: 7 days, ~200 miles, ~$200 receipts (52.8% error)
- **High receipt outliers**: Apply caps or corrections for >$2300 receipts
- **Long trip adjustments**: Refine predictions for 10+ day trips

## What NOT to Do
- Don't apply blanket corrections by trip type (made things 7.3% worse)
- Don't add more clusters without evidence
- Don't overcomplicate - v5 is already very good

## Expected Result
- Target MAE: ~$70-72 (7-10% improvement)
- Would maintain v5's robustness while fixing specific weaknesses
- Focus on precision for the 285 cases with >$100 error 