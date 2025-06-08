#!/usr/bin/env python3
"""
Symbolic Regression Hunt (R3)
- Fit symbolic regression to residuals after greedy rules
- Output sym_terms.py and diff plot
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_absolute_error

ERRORS_CSV = 'analysis/errors.csv'
RULES_JSON = 'analysis/greedy_rules.json'
SYM_TERMS_PY = 'analysis/sym_terms.py'
DIFF_PLOT = 'analysis/sym_diff_plot.png'

# Load errors and rules
df = pd.read_csv(ERRORS_CSV)
with open(RULES_JSON) as f:
    rules = json.load(f)

# Apply greedy rules
def apply_rules(df, rules):
    df = df.copy()
    for rule in rules:
        mask = (df['days_bucket'] == rule['days_bucket']) & (df['receipts_band'] == rule['receipts_band'])
        df.loc[mask, 'predicted'] += rule['bias']
    df['residual'] = df['predicted'] - df['expected']
    return df

df = apply_rules(df, rules)

# Features for symbolic regression
X = df[['days', 'miles', 'receipts', 'receipt_ratio', 'miles_per_day']].values
residuals = df['residual'].values

# Fit symbolic regressor
est = SymbolicRegressor(
    population_size=1000,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=42,
    function_set=('add', 'sub', 'mul', 'div', 'min', 'max'),
    max_depth=3
)
est.fit(X, residuals)

# Predict and evaluate
sym_pred = est.predict(X)
mae_before = mean_absolute_error(residuals, np.zeros_like(residuals))
mae_after = mean_absolute_error(residuals, sym_pred)
print(f"Symbolic regression MAE: before=${mae_before:.2f}, after=${mae_after:.2f}, improvement=${mae_before-mae_after:.2f}")

# Save symbolic terms as Python code
with open(SYM_TERMS_PY, 'w') as f:
    f.write('# Symbolic regression terms (auto-generated)\n')
    f.write('def symbolic_correction(days, miles, receipts, receipt_ratio, miles_per_day):\n')
    f.write('    """Symbolic regression correction for residuals after greedy rules."""\n')
    f.write(f'    return {est._program}\n')
print(f"Saved symbolic terms to {SYM_TERMS_PY}")

# Plot diff
plt.figure(figsize=(8,4))
plt.scatter(sym_pred, residuals, alpha=0.3, s=10)
plt.xlabel('Symbolic prediction')
plt.ylabel('Residual (after greedy rules)')
plt.title('Symbolic Regression Fit to Residuals')
plt.grid(True)
plt.savefig(DIFF_PLOT)
print(f"Saved diff plot to {DIFF_PLOT}") 