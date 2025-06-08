"""
Final Summary of All 6 Clusters
Consolidating findings for each calculation path
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

print("=" * 80)
print("FINAL SUMMARY: 6 CALCULATION PATHS")
print("=" * 80)

# Load cluster data
df = pd.read_csv(PROCESSED_DATA_DIR / 'public_cases_with_clusters.csv')

# Define our findings for each cluster
cluster_findings = {
    0: {
        "name": "Standard Multi-Day (5-12 days)",
        "size": 276,
        "characteristics": {
            "trip_days": "5-12 days (mode: 5)",
            "miles": "Low (avg 247)",
            "receipts": "Medium (avg $656)",
            "output": "~$1,012"
        },
        "formula": "output = 57.80 + 46.69*days + 0.51*miles + 0.71*receipts",
        "mae": 102.21,
        "notes": "Most common trip type, linear relationship works well"
    },
    
    1: {
        "name": "Single Day High Miles",
        "size": 50,
        "characteristics": {
            "trip_days": "1 day only",
            "miles": "High (avg 758)",
            "receipts": "High (avg $1,623)",
            "output": "~$1,258"
        },
        "formula": "Decision tree (complex rules)",
        "mae": 71.68,
        "notes": "Single day trips with high mileage, possibly air travel"
    },
    
    2: {
        "name": "Long Trip High Receipts",
        "size": 294,
        "characteristics": {
            "trip_days": "10-12 days (mode: 12)",
            "miles": "Medium-high (avg 732)",
            "receipts": "Very high (avg $1,826)",
            "output": "~$1,789"
        },
        "formula": "Decision tree (complex rules)",
        "mae": 101.28,
        "notes": "Extended trips with high expenses"
    },
    
    3: {
        "name": "Short Trip (3-5 days)",
        "size": 172,
        "characteristics": {
            "trip_days": "3-5 days (mode: 5)",
            "miles": "Medium (avg 560)",
            "receipts": "Very high (avg $1,732)",
            "output": "~$1,404"
        },
        "formula": "Decision tree (complex rules)",
        "mae": 88.65,
        "notes": "Short business trips with high daily expenses"
    },
    
    4: {
        "name": "Outlier (Low Receipt)",
        "size": 1,
        "characteristics": {
            "trip_days": "3 days",
            "miles": "93",
            "receipts": "$1.42",
            "output": "$365"
        },
        "formula": "output = 365 (fixed)",
        "mae": 0,
        "notes": "Single outlier case, possibly data error or special circumstance"
    },
    
    5: {
        "name": "Medium Trip High Miles (CONTAINS SPECIAL PROFILE)",
        "size": 207,
        "characteristics": {
            "trip_days": "5-8 days (mode: 5)",
            "miles": "High (avg 867)",
            "receipts": "Low-medium (avg $551)",
            "output": "~$1,154"
        },
        "formula": "Decision tree with special rule for 7 cases",
        "mae": 118.54,
        "special_profile": {
            "criteria": "7-8 days AND 900-1200 miles AND 1000-1200 receipts",
            "cases": 7,
            "output": "$2,015 - $2,280 (avg $2,126)",
            "pattern": "Step function based on receipt bins (MAE: $32.80)"
        },
        "notes": "Contains Kevin's special profile cases"
    }
}

# Print detailed summary
for cluster_id, info in cluster_findings.items():
    print(f"\n{'=' * 60}")
    print(f"CLUSTER {cluster_id}: {info['name']}")
    print(f"Size: {info['size']} cases ({info['size']/10:.1f}% of data)")
    print(f"{'=' * 60}")
    
    print("\nCharacteristics:")
    for key, value in info['characteristics'].items():
        print(f"  {key}: {value}")
    
    print(f"\nBest Formula: {info['formula']}")
    print(f"MAE: ${info['mae']:.2f}")
    
    if 'special_profile' in info:
        print("\n*** SPECIAL PROFILE ***")
        sp = info['special_profile']
        print(f"  Criteria: {sp['criteria']}")
        print(f"  Cases: {sp['cases']}")
        print(f"  Output range: {sp['output']}")
        print(f"  Pattern: {sp['pattern']}")
    
    print(f"\nNotes: {info['notes']}")

# Overall summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

total_mae = sum(info['mae'] * info['size'] for info in cluster_findings.values())
total_cases = sum(info['size'] for info in cluster_findings.values())
weighted_mae = total_mae / total_cases

print(f"\nTotal cases: {total_cases}")
print(f"Number of clusters: 6")
print(f"Weighted average MAE: ${weighted_mae:.2f}")
print(f"Improvement from v0.4: ${167.40 - weighted_mae:.2f} ({(167.40 - weighted_mae) / 167.40 * 100:.1f}%)")

print("\nKey Insights:")
print("1. Kevin was right - there are 6 distinct calculation paths")
print("2. Each cluster represents a different type of business trip")
print("3. The special profile (7 cases) is in Cluster 5")
print("4. Decision trees work better than linear models for most clusters")
print("5. The special profile cases follow a step function based on receipt amounts")

# Save summary to file
summary_data = []
for cluster_id, info in cluster_findings.items():
    summary_data.append({
        'cluster_id': cluster_id,
        'name': info['name'],
        'size': info['size'],
        'percentage': f"{info['size']/10:.1f}%",
        'mae': info['mae'],
        'formula_type': 'linear' if 'linear' in info['formula'].lower() else 'decision_tree' if 'tree' in info['formula'].lower() else 'fixed',
        'contains_special': 'special_profile' in info
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(REPORTS_DIR / 'final_cluster_summary.csv', index=False)
print(f"\nSaved final summary to '{REPORTS_DIR / 'final_cluster_summary.csv'}'")

# Create a simple visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Cluster sizes
sizes = [info['size'] for info in cluster_findings.values()]
names = [f"C{i}: {info['name'][:20]}..." for i, info in cluster_findings.items()]
colors = ['red' if i == 5 else 'skyblue' for i in range(6)]

ax1.bar(range(6), sizes, color=colors)
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Number of Cases')
ax1.set_title('Cluster Sizes (Red = Contains Special Profile)')
ax1.set_xticks(range(6))
ax1.set_xticklabels([f'C{i}' for i in range(6)])

# Add value labels
for i, v in enumerate(sizes):
    ax1.text(i, v + 5, str(v), ha='center')

# MAE by cluster
maes = [info['mae'] for info in cluster_findings.values()]
ax2.bar(range(6), maes, color=colors)
ax2.set_xlabel('Cluster')
ax2.set_ylabel('MAE ($)')
ax2.set_title('Model Error by Cluster')
ax2.set_xticks(range(6))
ax2.set_xticklabels([f'C{i}' for i in range(6)])

# Add value labels
for i, v in enumerate(maes):
    ax2.text(i, v + 2, f'${v:.0f}', ha='center')

# Add horizontal line for overall MAE
ax2.axhline(y=weighted_mae, color='green', linestyle='--', label=f'Overall MAE: ${weighted_mae:.0f}')
ax2.axhline(y=167.40, color='red', linestyle='--', label='v0.4 MAE: $167')
ax2.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'final_cluster_summary.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to '{FIGURES_DIR / 'final_cluster_summary.png'}'") 