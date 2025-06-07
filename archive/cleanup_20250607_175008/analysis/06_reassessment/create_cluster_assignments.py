"""
Create cluster assignments based on v0.5 model logic
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# Load public cases from JSON
with open(DATA_DIR / 'raw' / 'public_cases.json', 'r') as f:
    data = json.load(f)

# Create DataFrame
cases = []
for case in data:
    cases.append({
        'trip_days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'],
        'receipts': case['input']['total_receipts_amount'],
        'expected_output': case['expected_output']
    })
df = pd.DataFrame(cases)

# Function to assign clusters based on v0.5 logic
def assign_cluster(row):
    days = row['trip_days']
    miles = row['miles']
    receipts = row['receipts']
    
    # Check for special profile first (Cluster 5)
    if 7 <= days <= 8 and 900 <= miles <= 1200 and 1000 <= receipts <= 1200:
        return 5
    
    # Cluster 4: Outlier (very low receipts)
    if receipts < 10:
        return 4
        
    # Cluster 1: Single day high miles  
    if days == 1 and miles > 600:
        return 1
        
    # Cluster 2: Long trip (10+ days) with high receipts
    if days >= 10 and receipts > 1500:
        return 2
        
    # Cluster 3: Short trip (3-5 days) with very high expenses
    if 3 <= days <= 5 and receipts > 1700:
        return 3
        
    # Cluster 5: Medium trip (5-8 days) with high miles
    if 5 <= days <= 8 and miles > 800:
        return 5
        
    # Default to Cluster 0: Standard multi-day
    return 0

# Assign clusters
df['cluster'] = df.apply(assign_cluster, axis=1)

# Print distribution
print("Cluster distribution:")
for cluster in sorted(df['cluster'].unique()):
    count = (df['cluster'] == cluster).sum()
    pct = count / len(df) * 100
    print(f"Cluster {cluster}: {count} cases ({pct:.1f}%)")

# Save with clusters
df.to_csv(DATA_DIR / 'public_cases_with_clusters.csv', index=False)
print(f"\nSaved to: {DATA_DIR / 'public_cases_with_clusters.csv'}") 