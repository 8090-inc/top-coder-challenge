#!/usr/bin/env python3
"""
Greedy Rule Mining (R2)
- Iteratively patch largest error buckets in errors.csv with simple additive/linear rules
- Stop when global exact-match >94% or no more large-error buckets
- Output greedy_rules.json and bias_log.md
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

ERRORS_CSV = 'analysis/errors.csv'
RULES_JSON = 'analysis/greedy_rules.json'
BIAS_LOG = 'analysis/bias_log.md'

# Load errors
orig_err = pd.read_csv(ERRORS_CSV)
err = orig_err.copy()

# Track rules and patched buckets
rules = []
log_lines = []
patched_buckets = set()

# Helper: apply all rules to a DataFrame
# Each rule: {'days_bucket': str, 'receipts_band': int, 'bias': float}
def apply_rules(df, rules):
    df = df.copy()
    for rule in rules:
        mask = (df['days_bucket'] == rule['days_bucket']) & (df['receipts_band'] == rule['receipts_band'])
        df.loc[mask, 'predicted'] += rule['bias']
    df['error'] = df['predicted'] - df['expected']
    df['abs_error'] = df['error'].abs()
    df['exact'] = (df['abs_error'] <= 0.01)
    return df

# Main loop
while True:
    # Always start from original predictions
    err = apply_rules(orig_err, rules)
    # Compute global stats
    mae = err['abs_error'].mean()
    exact_pct = 100 * err['exact'].mean()
    print(f"Current: MAE=${mae:.2f}, Exact-match={exact_pct:.2f}% ({err['exact'].sum()}/{len(err)})")
    log_lines.append(f"MAE=${mae:.2f}, Exact-match={exact_pct:.2f}% ({err['exact'].sum()}/{len(err)})")
    if exact_pct > 94:
        print("Target reached!")
        break
    # Find largest error bucket (mean error >$20, ≥10 cases, not already patched)
    gb = err.groupby(['days_bucket', 'receipts_band'])
    stats = gb.agg(count=('error', 'size'), mean_error=('error', 'mean'))
    stats = stats[(stats['count'] >= 10) & (stats['mean_error'].abs() > 20)]
    # Exclude already patched buckets
    stats = stats[~stats.index.isin(patched_buckets)]
    if stats.empty:
        print("No more large-error buckets.")
        break
    # Pick the bucket with largest |mean_error|
    idx = stats['mean_error'].abs().idxmax()
    days_bucket, receipts_band = idx
    mean_err = stats.loc[idx, 'mean_error']
    count = stats.loc[idx, 'count']
    # Fit a simple additive bias (negate mean error)
    bias = -float(mean_err)
    rule = {
        'days_bucket': str(days_bucket),
        'receipts_band': int(receipts_band),
        'bias': float(bias)
    }
    rules.append(rule)
    patched_buckets.add(idx)
    log_lines.append(f"Patched bucket (days_bucket={days_bucket}, receipts_band={receipts_band}, count={count}): bias={bias:+.2f}")
    print(f"Patched bucket (days_bucket={days_bucket}, receipts_band={receipts_band}, count={count}): bias={bias:+.2f}")

# Save rules and log
with open(RULES_JSON, 'w') as f:
    json.dump(rules, f, indent=2)
with open(BIAS_LOG, 'w') as f:
    f.write('\n'.join(log_lines))
print(f"Saved {len(rules)} rules to {RULES_JSON}") 