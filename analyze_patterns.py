import json
import pandas as pd
import numpy as np

# Load the data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame for easier analysis
df = pd.DataFrame([
    {
        'days': item['input']['trip_duration_days'],
        'miles': item['input']['miles_traveled'],
        'receipts': item['input']['total_receipts_amount'],
        'output': item['expected_output']
    }
    for item in data
])

# Add calculated columns
df['miles_per_day'] = df['miles'] / df['days']
df['receipts_ends_49'] = df['receipts'].apply(lambda x: str(x).endswith('.49'))
df['receipts_ends_99'] = df['receipts'].apply(lambda x: str(x).endswith('.99'))
df['receipts_special'] = df['receipts_ends_49'] | df['receipts_ends_99']

print("=== ANALYSIS OF TRAVEL REIMBURSEMENT PATTERNS ===\n")

# 1. Base Per Diem Analysis
print("1. BASE PER DIEM ANALYSIS")
print("-" * 50)
for days in range(1, 8):
    trips = df[df['days'] == days]
    if len(trips) > 0:
        # Find trips with minimal miles and receipts to isolate base per diem
        simple_trips = trips[(trips['miles'] < 50) & (trips['receipts'] < 30)]
        if len(simple_trips) > 0:
            print(f"{days}-day trips: {len(trips)} total, {len(simple_trips)} simple")
            print(f"  Simple trip outputs: {simple_trips['output'].min():.2f} - {simple_trips['output'].max():.2f}")
            print(f"  Estimated base per diem: {simple_trips['output'].min() / days:.2f}")
        else:
            print(f"{days}-day trips: {len(trips)} total, no simple trips found")
print()

# 2. 5-Day Trip Bonus Analysis
print("2. 5-DAY TRIP BONUS ANALYSIS")
print("-" * 50)
five_day = df[df['days'] == 5]
print(f"Total 5-day trips: {len(five_day)}")
print(f"Output range: ${five_day['output'].min():.2f} - ${five_day['output'].max():.2f}")
# Compare with expected base (500 = 5 * 100)
low_receipt_five = five_day[five_day['receipts'] < 100]
if len(low_receipt_five) > 0:
    print(f"Low-receipt 5-day trips suggest bonus present (outputs > $500 base)")
    print(f"  Examples: {low_receipt_five[['receipts', 'output']].head(3).to_dict('records')}")
print()

# 3. Mileage Rate Analysis
print("3. MILEAGE RATE TIERS")
print("-" * 50)
# Analyze 1-day trips to isolate mileage component
one_day = df[(df['days'] == 1) & (df['receipts'] < 30)].sort_values('miles')
if len(one_day) > 5:
    print("Analyzing 1-day trips with low receipts:")
    # Calculate approximate mileage reimbursement
    one_day['approx_mileage_reimb'] = one_day['output'] - 100 - one_day['receipts']
    one_day['rate_per_mile'] = one_day['approx_mileage_reimb'] / one_day['miles']
    
    # Look for tier breakpoints
    for threshold in [50, 100, 150, 200]:
        below = one_day[one_day['miles'] < threshold]
        above = one_day[one_day['miles'] >= threshold]
        if len(below) > 0 and len(above) > 0:
            print(f"  Miles < {threshold}: avg rate ${below['rate_per_mile'].mean():.3f}/mile")
            print(f"  Miles >= {threshold}: avg rate ${above['rate_per_mile'].mean():.3f}/mile")
print()

# 4. Efficiency Bonus Analysis
print("4. EFFICIENCY BONUS (HIGH MILES/DAY)")
print("-" * 50)
high_efficiency = df[df['miles_per_day'] > 100]
print(f"Trips with >100 miles/day: {len(high_efficiency)}")
if len(high_efficiency) > 0:
    print("Examples:")
    for _, trip in high_efficiency.head(5).iterrows():
        print(f"  {trip['days']} days, {trip['miles']} miles ({trip['miles_per_day']:.1f}/day): ${trip['output']:.2f}")
print()

# 5. Receipt Patterns
print("5. RECEIPT PATTERNS")
print("-" * 50)
print(f"Receipts ending in .49: {df['receipts_ends_49'].sum()}")
print(f"Receipts ending in .99: {df['receipts_ends_99'].sum()}")
special_receipts = df[df['receipts_special']]
if len(special_receipts) > 0:
    print("Examples of .49/.99 receipts:")
    for _, trip in special_receipts.head(5).iterrows():
        print(f"  ${trip['receipts']:.2f} → ${trip['output']:.2f}")
print()

# 6. Low Receipt Penalty Analysis
print("6. LOW RECEIPT PENALTIES")
print("-" * 50)
low_receipts = df[df['receipts'] < 10]
print(f"Trips with receipts < $10: {len(low_receipts)}")
if len(low_receipts) > 0:
    print("Examples showing potential penalties:")
    for _, trip in low_receipts.head(5).iterrows():
        expected_min = trip['days'] * 100  # minimum expected without penalty
        print(f"  {trip['days']} days, ${trip['receipts']:.2f} receipts: ${trip['output']:.2f} (base would be ${expected_min})")
print()

# 7. Receipt Caps Analysis
print("7. RECEIPT REIMBURSEMENT CAPS")
print("-" * 50)
# Look for patterns where high receipts don't proportionally increase output
high_receipts = df[df['receipts'] > 1000].sort_values('receipts')
if len(high_receipts) > 0:
    print("High receipt examples:")
    for _, trip in high_receipts.head(5).iterrows():
        receipt_portion = trip['output'] - (trip['days'] * 100)  # subtract base per diem
        print(f"  ${trip['receipts']:.2f} receipts → ${trip['output']:.2f} total (receipt portion ≈ ${receipt_portion:.2f})")