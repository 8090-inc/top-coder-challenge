#!/usr/bin/env python3
import json
import sys

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print(f"Total cases: {len(cases)}")

# Analyze basic patterns
print("\n=== BASIC PATTERNS ===")

# Look at per-day base rates
per_day_rates = {}
for case in cases[:100]:  # First 100 cases
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    if days not in per_day_rates:
        per_day_rates[days] = []
    
    # Rough estimate: if no miles and no receipts, what would the output be?
    # This is hard to estimate, so let's just see base outputs
    per_day_rates[days].append(output)

for days in sorted(per_day_rates.keys()):
    outputs = per_day_rates[days]
    avg = sum(outputs) / len(outputs)
    print(f"{days} days: avg={avg:.2f}, samples={len(outputs)}")

print("\n=== 5-DAY BONUS CHECK ===")
five_day_cases = [c for c in cases if c['input']['trip_duration_days'] == 5]
print(f"5-day cases: {len(five_day_cases)}")

# Look for efficiency patterns
print("\n=== EFFICIENCY PATTERNS ===")
efficiency_cases = []
for case in cases[:200]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    mpd = miles / days
    efficiency_cases.append((mpd, output, days, miles, receipts))

efficiency_cases.sort()
print("MPD ranges and typical outputs:")
for i in range(0, len(efficiency_cases), 20):
    mpd, output, days, miles, receipts = efficiency_cases[i]
    print(f"MPD {mpd:.1f}: output={output:.2f} (days={days}, miles={miles}, receipts=${receipts:.2f})")

print("\n=== RECEIPT WINDFALL CHECK ===")
windfall_49 = [c for c in cases if str(c['input']['total_receipts_amount']).endswith('.49')]
windfall_99 = [c for c in cases if str(c['input']['total_receipts_amount']).endswith('.99')]
normal = [c for c in cases if not (str(c['input']['total_receipts_amount']).endswith('.49') or str(c['input']['total_receipts_amount']).endswith('.99'))]

print(f"Cases ending in .49: {len(windfall_49)}")
print(f"Cases ending in .99: {len(windfall_99)}")
print(f"Normal cases: {len(normal)}")

if windfall_49:
    print("Sample .49 cases:")
    for case in windfall_49[:5]:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        output = case['expected_output']
        print(f"  {days}d, {miles}mi, ${receipts} -> ${output}") 