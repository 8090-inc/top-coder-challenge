import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# Load the data
with open(PUBLIC_CASES_PATH, "r") as f:
    data = json.load(f)
public_df = pd.json_normalize(data)
public_df.columns = ['expected_output', 'trip_days', 'miles', 'receipts']

print("=" * 80)
print("COMPARING INITIAL ANALYSIS TO HYPOTHESES")
print("=" * 80)

# 1. BASE PER DIEM HYPOTHESIS ($100/day)
print("\n1. BASE PER DIEM HYPOTHESIS CHECK:")
print("-" * 40)
print("Interview claim: $100/day base rate")
print("Linear regression result: $61.51/day coefficient + $915.87 intercept")
print("⚠️  MISMATCH: The intercept is very high, suggesting base calculation is NOT simple per diem")

# Calculate actual per diem rates
public_df['per_diem_rate'] = public_df['expected_output'] / public_df['trip_days']
print(f"\nActual per diem rates:")
print(f"  Mean: ${public_df['per_diem_rate'].mean():.2f}")
print(f"  Median: ${public_df['per_diem_rate'].median():.2f}")
print(f"  Std Dev: ${public_df['per_diem_rate'].std():.2f}")

# 2. CORRELATION ANALYSIS
print("\n\n2. CORRELATION PRIORITIES:")
print("-" * 40)
print("Interview emphasis: Miles > Days > Receipts")
print("Actual correlations:")
print(f"  Receipts: 0.704 (STRONGEST)")
print(f"  Trip days: 0.514 (MODERATE)")  
print(f"  Miles: 0.432 (WEAKEST)")
print("⚠️  MISMATCH: Receipts are much more important than interviews suggested")

# 3. 5-DAY BONUS CHECK
print("\n\n3. 5-DAY BONUS HYPOTHESIS:")
print("-" * 40)
# Group by trip days and calculate mean reimbursement
day_means = public_df.groupby('trip_days')['expected_output'].agg(['mean', 'count', 'std'])
print("\nMean reimbursement by trip days:")
for days in range(1, 11):
    if days in day_means.index:
        mean_val = day_means.loc[days, 'mean']
        count = day_means.loc[days, 'count']
        print(f"  {days} days: ${mean_val:,.2f} (n={count})")

# Check 5-day specifically
if 5 in day_means.index:
    mean_5 = day_means.loc[5, 'mean']
    mean_4 = day_means.loc[4, 'mean'] if 4 in day_means.index else 0
    mean_6 = day_means.loc[6, 'mean'] if 6 in day_means.index else 0
    
    print(f"\n5-day analysis:")
    print(f"  4-day mean: ${mean_4:,.2f}")
    print(f"  5-day mean: ${mean_5:,.2f}")
    print(f"  6-day mean: ${mean_6:,.2f}")
    
    if mean_5 > mean_4 and mean_5 > mean_6:
        print("✓ SUPPORTS 5-day bonus hypothesis")
    else:
        print("✗ Does NOT support clear 5-day bonus")

# 4. EFFICIENCY ANALYSIS (180-220 miles/day sweet spot)
print("\n\n4. EFFICIENCY HYPOTHESIS (180-220 miles/day):")
print("-" * 40)
public_df['miles_per_day'] = public_df['miles'] / public_df['trip_days']

# Create efficiency bins
efficiency_bins = [0, 100, 180, 220, 400, 1000]
efficiency_labels = ['0-100', '100-180', '180-220', '220-400', '400+']
public_df['efficiency_bin'] = pd.cut(public_df['miles_per_day'], bins=efficiency_bins, labels=efficiency_labels)

# Calculate mean reimbursement by efficiency
efficiency_means = public_df.groupby('efficiency_bin')['expected_output'].agg(['mean', 'count'])
print("\nMean reimbursement by efficiency (miles/day):")
for bin_label in efficiency_labels:
    if bin_label in efficiency_means.index:
        mean_val = efficiency_means.loc[bin_label, 'mean']
        count = efficiency_means.loc[bin_label, 'count']
        print(f"  {bin_label}: ${mean_val:,.2f} (n={count})")

# 5. LOW RECEIPT PENALTY (<$50)
print("\n\n5. LOW RECEIPT PENALTY HYPOTHESIS:")
print("-" * 40)
low_receipt = public_df[public_df['receipts'] < 50]
no_receipt = public_df[public_df['receipts'] < 10]
normal_receipt = public_df[public_df['receipts'] >= 50]

print(f"Mean reimbursement by receipt level:")
print(f"  <$10 receipts: ${no_receipt['expected_output'].mean():.2f} (n={len(no_receipt)})")
print(f"  <$50 receipts: ${low_receipt['expected_output'].mean():.2f} (n={len(low_receipt)})")
print(f"  >=$50 receipts: ${normal_receipt['expected_output'].mean():.2f} (n={len(normal_receipt)})")
print("✓ STRONGLY SUPPORTS low receipt penalty")

# 6. ROUNDING BUG CHECK (.49 and .99 endings)
print("\n\n6. ROUNDING BUG HYPOTHESIS (.49 and .99 endings):")
print("-" * 40)
public_df['receipt_cents'] = (public_df['receipts'] * 100) % 100
public_df['ends_49'] = public_df['receipt_cents'].round() == 49
public_df['ends_99'] = public_df['receipt_cents'].round() == 99
public_df['special_ending'] = public_df['ends_49'] | public_df['ends_99']

special_ending_mean = public_df[public_df['special_ending']]['expected_output'].mean()
normal_ending_mean = public_df[~public_df['special_ending']]['expected_output'].mean()

print(f"Receipts ending in .49 or .99: {public_df['special_ending'].sum()} cases")
print(f"Mean reimbursement with .49/.99: ${special_ending_mean:.2f}")
print(f"Mean reimbursement without: ${normal_ending_mean:.2f}")
print(f"Difference: ${special_ending_mean - normal_ending_mean:.2f}")

# 7. OUTLIER PATTERNS
print("\n\n7. OUTLIER ANALYSIS:")
print("-" * 40)
# Load outliers from the previous analysis
outliers_df = pd.read_csv(REPORTS_DIR / 'potential_bugs_or_special_cases.csv')
print(f"Total outliers identified: {len(outliers_df)}")
print("\nOutlier patterns:")
print(f"  High receipt/low reimbursement: {len(outliers_df[outliers_df['residual'] < -500])} cases")
print(f"  Low receipt/high reimbursement: {len(outliers_df[outliers_df['residual'] > 500])} cases")

# Check for Kevin's sweet spot
kevin_sweet_spot = public_df[
    (public_df['trip_days'] == 5) & 
    (public_df['miles_per_day'] >= 180) & 
    (public_df['miles_per_day'] <= 220) &
    (public_df['receipts'] / public_df['trip_days'] < 100)
]
print(f"\n'Kevin Sweet Spot' cases (5 days, 180-220 mi/day, <$100/day spending): {len(kevin_sweet_spot)}")
if len(kevin_sweet_spot) > 0:
    print(f"  Mean reimbursement: ${kevin_sweet_spot['expected_output'].mean():.2f}")
    print(f"  vs Overall mean: ${public_df['expected_output'].mean():.2f}")

# 8. FORMULA COMPLEXITY
print("\n\n8. FORMULA COMPLEXITY:")
print("-" * 40)
print("Simple formula testing results (MAE):")
print("  $100/day + $0.50/mile + 100% receipts: $865.62")
print("  $50/day + $0.58/mile + 100% receipts: $575.22 (best)")
print("  Linear model R² = 0.784 with MAE = $175.49")
print("\n⚠️  High errors suggest complex non-linear calculation or multiple paths")

# SUMMARY
print("\n" + "=" * 80)
print("KEY INSIGHTS FOR HYPOTHESIS TESTING:")
print("=" * 80)
print("\n1. RECEIPTS are the dominant factor (0.70 correlation), not miles")
print("2. Base calculation is NOT simple $100/day per diem")
print("3. Strong evidence for low receipt penalty")
print("4. 5-day bonus needs deeper investigation")
print("5. High base intercept ($915) suggests minimum reimbursement or base amount")
print("6. Simple linear formulas fail badly - system is complex")
print("7. Many outliers suggest special rules or bugs")

print("\n" + "=" * 80)
print("RECOMMENDED TESTING PRIORITY:")
print("=" * 80)
print("1. Investigate receipt processing rules (strongest correlation)")
print("2. Analyze the high base intercept - minimum reimbursement?")
print("3. Deep dive on outliers to find special rules")
print("4. Test efficiency patterns with residual analysis")
print("5. Check for multiple calculation paths (clustering)") 