#!/usr/bin/env python3
"""
Deep Differential Audit (R1)
- Run GBDT model on all public and private cases
- Output errors.csv with all required features and error columns
- Generate error_heatmap.png (mean error by days_bucket x receipts_band)
- Print top-25 error hotspots
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from final_calculator import calculate_reimbursement_final

ROOT = Path(__file__).parent.parent
DATA_FILES = [
    (ROOT / 'public_cases.json', 'public'),
    (ROOT / 'private_cases.json', 'private'),
]

# Bucketing functions
def days_bucket(days):
    if days <= 4:
        return '≤4'
    elif days == 5:
        return '5'
    elif days <= 8:
        return '6–8'
    else:
        return '≥9'

def receipts_band(receipts):
    return int(receipts // 250) * 250

def miles_band(miles):
    return int(miles // 200) * 200

def main():
    records = []
    trip_id = 0
    for file, source in DATA_FILES:
        with open(file) as f:
            data = json.load(f)
        for row in data:
            # Only analyze public cases with ground truth
            if 'input' in row and 'expected_output' in row:
                days = float(row['input']['trip_duration_days'])
                miles = float(row['input']['miles_traveled'])
                receipts = float(row['input']['total_receipts_amount'])
                expected = float(row['expected_output'])
            else:
                print(f"[INFO] Skipping row without ground truth: {row}")
                continue
            pred = calculate_reimbursement_final(days, miles, receipts)
            error = pred - expected
            rec_ratio = receipts / days if days else 0.0
            mpd = miles / days if days else 0.0
            records.append({
                'trip_id': trip_id,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': pred,
                'error': error,
                'abs_error': abs(error),
                'days_bucket': days_bucket(days),
                'receipts_band': receipts_band(receipts),
                'miles_band': miles_band(miles),
                'receipt_ratio': rec_ratio,
                'miles_per_day': mpd,
                'source': source,
            })
            trip_id += 1
    df = pd.DataFrame(records)
    out_csv = Path(__file__).parent / 'errors.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved errors.csv with {len(df)} rows")

    # Heatmap: mean abs error by days_bucket x receipts_band
    pivot = df.pivot_table(index='days_bucket', columns='receipts_band', values='abs_error', aggfunc='mean')
    plt.figure(figsize=(16, 6))
    plt.title('Mean Absolute Error by Days Bucket and Receipts Band')
    plt.xlabel('Receipts Band ($)')
    plt.ylabel('Days Bucket')
    im = plt.imshow(pivot, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Mean Absolute Error ($)')
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.tight_layout()
    heatmap_path = Path(__file__).parent / 'error_heatmap.png'
    plt.savefig(heatmap_path)
    print(f"Saved error_heatmap.png")

    # Top-25 error hotspots (bucket-feature combos with highest mean error, ≥10 cases)
    group = df.groupby(['days_bucket', 'receipts_band']).agg(
        count=('error', 'size'),
        mean_error=('abs_error', 'mean')
    ).reset_index()
    hotspots = group[group['count'] >= 10].sort_values('mean_error', ascending=False).head(25)
    print("\nTop-25 Error Hotspots (≥10 cases):")
    print(hotspots[['days_bucket', 'receipts_band', 'count', 'mean_error']].to_string(index=False))

if __name__ == '__main__':
    main() 