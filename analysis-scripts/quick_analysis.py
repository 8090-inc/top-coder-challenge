#!/usr/bin/env python3
import json
import statistics

def analyze_cases():
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    print("=== BASIC PATTERNS ANALYSIS ===")
    print("Sample cases (first 15):")
    print("Days\tMiles\tReceipts\tOutput\t\t$/Day")
    print("-" * 55)
    
    for case in cases[:15]:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        per_day = o / d
        print(f"{d}\t{m}\t${r:.2f}\t\t${o:.2f}\t\t${per_day:.2f}")
    
    # Analyze per-day rates
    per_day_rates = []
    for case in cases[:100]:
        d = case['input']['trip_duration_days']
        o = case['expected_output']
        per_day_rates.append(o / d)
    
    print(f"\n=== PER-DAY RATE ANALYSIS (first 100 cases) ===")
    print(f"Range: ${min(per_day_rates):.2f} - ${max(per_day_rates):.2f} per day")
    print(f"Average: ${statistics.mean(per_day_rates):.2f} per day")
    print(f"Median: ${statistics.median(per_day_rates):.2f} per day")
    
    # Look for base rate patterns
    print(f"\n=== BASE RATE PATTERNS ===")
    # Simple cases with low receipts
    simple_cases = [c for c in cases[:200] if c['input']['total_receipts_amount'] < 25][:10]
    print("Simple cases (low receipts):")
    for case in simple_cases:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        print(f"{d}d, {m}mi, ${r:.2f} -> ${o:.2f}")
        
        # Try to deduce base components
        estimated_per_day = 100  # Guess from interviews
        estimated_mileage = m * 0.5  # Rough guess
        base_estimate = (d * estimated_per_day) + estimated_mileage + r
        print(f"  Rough estimate: ${base_estimate:.2f} (diff: ${o - base_estimate:.2f})")
    
if __name__ == "__main__":
    analyze_cases() 