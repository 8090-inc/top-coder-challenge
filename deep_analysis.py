#!/usr/bin/env python3
import json
import statistics
from collections import defaultdict

def analyze_comprehensive():
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    print("=== COMPREHENSIVE ANALYSIS ===")
    
    # 1. Analyze base per-day patterns
    print("\n=== PER-DAY ANALYSIS BY TRIP LENGTH ===")
    by_days = defaultdict(list)
    for case in cases:
        d = case['input']['trip_duration_days']
        o = case['expected_output']
        by_days[d].append(o / d)
    
    for days in sorted(by_days.keys())[:10]:  # First 10 day lengths
        rates = by_days[days]
        if len(rates) >= 5:  # Only if we have enough samples
            print(f"{days} days: avg=${statistics.mean(rates):.2f}/day, "
                  f"median=${statistics.median(rates):.2f}/day, "
                  f"range=${min(rates):.2f}-${max(rates):.2f} ({len(rates)} cases)")
    
    # 2. Look at minimal cases (low receipts) to find base formula
    print("\n=== MINIMAL CASES (receipts < $10) ===")
    minimal_cases = [c for c in cases if c['input']['total_receipts_amount'] < 10][:20]
    print("Days\tMiles\tReceipts\tOutput\t\tBase$/day\tMiles$/day")
    
    for case in minimal_cases:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        base_per_day = o / d
        
        # Estimate mileage component
        estimated_base = d * 60  # Try $60/day base
        remaining = o - estimated_base
        miles_rate = remaining / m if m > 0 else 0
        
        print(f"{d}\t{m}\t${r:.2f}\t\t${o:.2f}\t\t${base_per_day:.2f}\t\t${miles_rate:.3f}")
    
    # 3. Analyze receipt patterns
    print("\n=== RECEIPT REIMBURSEMENT PATTERNS ===")
    print("Looking at cases grouped by receipt amounts...")
    
    receipt_groups = [
        (0, 50, "Low receipts ($0-50)"),
        (100, 200, "Medium receipts ($100-200)"),
        (500, 600, "High receipts ($500-600)"),
        (1000, 1500, "Very high receipts ($1000-1500)")
    ]
    
    for low, high, label in receipt_groups:
        group_cases = [c for c in cases if low <= c['input']['total_receipts_amount'] <= high][:10]
        if group_cases:
            print(f"\n{label}:")
            for case in group_cases:
                d = case['input']['trip_duration_days']
                m = case['input']['miles_traveled']
                r = case['input']['total_receipts_amount']
                o = case['expected_output']
                
                # Estimate non-receipt components
                est_base = d * 60
                est_miles = m * 0.3  # Rough guess
                est_receipts_component = o - est_base - est_miles
                receipt_rate = est_receipts_component / r if r > 0 else 0
                
                print(f"  {d}d, {m}mi, ${r:.2f} -> ${o:.2f} "
                      f"(receipt comp: ${est_receipts_component:.2f}, rate: {receipt_rate:.2f})")

if __name__ == "__main__":
    analyze_comprehensive() 