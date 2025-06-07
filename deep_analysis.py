#!/usr/bin/env python3
import json
import numpy as np
from collections import defaultdict

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== REVERSE ENGINEERING THE FORMULA ===\n")

# Find cases with minimal receipts to understand base + mileage
print("1. BASE + MILEAGE ANALYSIS (low receipt cases)")
low_receipt_cases = [c for c in cases if c['input']['total_receipts_amount'] < 30]
print(f"Cases with <$30 receipts: {len(low_receipt_cases)}")

for case in low_receipt_cases[:10]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Rough per-day calculation
    per_day = output / days
    # Rough per-mile calculation  
    per_mile = (output - (days * 100)) / miles if miles > 0 else 0
    
    print(f"  {days}d, {miles}mi, ${receipts:.2f} -> ${output:.2f} (${per_day:.2f}/day, ${per_mile:.3f}/mi)")

print("\n2. MILEAGE RATE ANALYSIS")
# Group by mileage ranges to see rate changes
mileage_groups = defaultdict(list)
for case in cases:
    miles = case['input']['miles_traveled']
    days = case['input']['trip_duration_days']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    if miles > 0:
        # Rough estimate: subtract $100/day base and see what's left for mileage
        base_estimate = days * 100
        if days == 5:
            base_estimate += 50  # 5-day bonus from interviews
        
        remaining = output - base_estimate - (0.8 * min(receipts, 600) + 0.2 * max(0, receipts - 600))
        rate = remaining / miles if remaining > 0 else 0
        
        if miles <= 100:
            mileage_groups['0-100'].append(rate)
        elif miles <= 500:
            mileage_groups['101-500'].append(rate)
        else:
            mileage_groups['500+'].append(rate)

for group, rates in mileage_groups.items():
    if rates:
        avg_rate = np.mean(rates)
        print(f"  {group} miles: avg rate ${avg_rate:.3f}/mile ({len(rates)} samples)")

print("\n3. 5-DAY BONUS ANALYSIS")
five_day = [c for c in cases if c['input']['trip_duration_days'] == 5]
four_day = [c for c in cases if c['input']['trip_duration_days'] == 4]
six_day = [c for c in cases if c['input']['trip_duration_days'] == 6]

def calc_per_day_premium(cases, label):
    if not cases:
        return
    premiums = []
    for case in cases[:10]:
        days = case['input']['trip_duration_days']
        output = case['expected_output']
        premium = output / days
        premiums.append(premium)
    avg_premium = np.mean(premiums)
    print(f"  {label}: ${avg_premium:.2f}/day average ({len(cases)} total cases)")

calc_per_day_premium(four_day, "4-day trips")
calc_per_day_premium(five_day, "5-day trips")  
calc_per_day_premium(six_day, "6-day trips")

print("\n4. EFFICIENCY BONUS ANALYSIS")
efficiency_bonuses = []
for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    if days > 0 and miles > 0:
        mpd = miles / days
        # Look for cases in Kevin's "sweet spot" (150-220 mpd)
        if 150 <= mpd <= 220:
            efficiency_bonuses.append((mpd, output, days, miles, receipts))

print(f"Cases in efficiency sweet spot (150-220 MPD): {len(efficiency_bonuses)}")
for mpd, output, days, miles, receipts in efficiency_bonuses[:5]:
    print(f"  MPD {mpd:.1f}: {days}d, {miles}mi, ${receipts:.2f} -> ${output:.2f}")

print("\n5. RECEIPT PROCESSING ANALYSIS")
# Look for cases with similar days/miles but different receipts
receipt_analysis = defaultdict(list)
for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Group by similar days/miles
    key = f"{days}d_{miles//50*50}mi"  # Round miles to nearest 50
    receipt_analysis[key].append((receipts, output))

print("Receipt impact (similar day/mile scenarios):")
for key, data in list(receipt_analysis.items())[:5]:
    if len(data) >= 2:
        data.sort()  # Sort by receipt amount
        print(f"  {key}:")
        for receipts, output in data[:3]:
            print(f"    ${receipts:.2f} receipts -> ${output:.2f}")

print("\n6. WINDFALL ANALYSIS (.49/.99 endings)")
windfall_49 = []
windfall_99 = []
normal_comparison = []

for case in cases:
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    
    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith('.49'):
        windfall_49.append((days, miles, receipts, output))
    elif receipt_str.endswith('.99'):
        windfall_99.append((days, miles, receipts, output))
    else:
        # Find similar cases for comparison
        similar_receipts = abs(receipts - round(receipts, 0)) < 0.1  # Near round numbers
        if similar_receipts and len(normal_comparison) < 20:
            normal_comparison.append((days, miles, receipts, output))

print(f".49 cases: {len(windfall_49)}")
print(f".99 cases: {len(windfall_99)}")
if windfall_49:
    print("Sample .49 cases:")
    for days, miles, receipts, output in windfall_49[:3]:
        print(f"  {days}d, {miles}mi, ${receipts} -> ${output}") 