#!/usr/bin/env python3
import json
import statistics

def analyze_multiday_failures():
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Focus on the specific failing cases from eval
    failing_cases_data = [
        (4, 69, 2321.49, 322.00),    # Case 152
        (8, 1025, 1031.33, 2214.64), # Case 513  
        (7, 1006, 1181.33, 2279.82), # Case 149
        (8, 795, 1645.99, 644.69),   # Case 684
        (7, 1033, 1013.03, 2119.83), # Case 669
    ]
    
    print("=== MULTI-DAY FAILURE ANALYSIS ===")
    print("Days\tMiles\tReceipts\tExpected\t$/Day\tMPD")
    print("-" * 65)
    
    for d, m, r, expected in failing_cases_data:
        per_day = expected / d
        mpd = m / d
        print(f"{d}\t{m}\t${r:.2f}\t\t${expected:.2f}\t\t${per_day:.2f}\t{mpd:.1f}")
    
    print("\n=== PATTERN ANALYSIS ===")
    
    # Group 1: High receipt, low expected (Cases 152, 684)
    print("\n1. HIGH RECEIPTS, LOW EXPECTED:")
    high_receipt_low = [(4, 69, 2321.49, 322.00), (8, 795, 1645.99, 644.69)]
    for d, m, r, expected in high_receipt_low:
        est_base = d * 50  # Rough estimate
        est_mileage = m * 0.3
        est_total_before_receipts = est_base + est_mileage
        remaining_for_receipts = expected - est_total_before_receipts
        receipt_rate = remaining_for_receipts / r if r > 0 else 0
        
        print(f"  {d}d, {m}mi, ${r:.2f} -> ${expected:.2f}")
        print(f"    Est base+miles: ${est_total_before_receipts:.2f}")
        print(f"    Receipt component: ${remaining_for_receipts:.2f} (rate: {receipt_rate:.3f})")
    
    # Group 2: High mileage, high expected (Cases 513, 149, 669)
    print("\n2. HIGH MILEAGE, HIGH EXPECTED:")
    high_mileage_high = [(8, 1025, 1031.33, 2214.64), (7, 1006, 1181.33, 2279.82), (7, 1033, 1013.03, 2119.83)]
    for d, m, r, expected in high_mileage_high:
        per_day = expected / d
        mpd = m / d
        
        # Our current formula
        base = d * 45  # 8-day rate
        mileage = m * 0.3
        if r < 100:
            net_receipts = r * 1.0
        elif r < 600:
            net_receipts = r * 0.8
        else:
            net_receipts = (600 * 0.8) + ((r - 600) * 0.4)
        
        our_estimate = base + mileage + net_receipts
        error = expected - our_estimate
        
        print(f"  {d}d, {m}mi, ${r:.2f} -> ${expected:.2f} (${per_day:.2f}/day, {mpd:.1f} MPD)")
        print(f"    Our estimate: ${our_estimate:.2f}, Error: ${error:.2f}")
        print(f"    Suggests mileage rate should be: ${(expected - base - net_receipts) / m:.3f}/mile")
    
    print("\n=== EFFICIENCY BONUS HYPOTHESIS ===")
    print("High MPD cases might get efficiency bonuses...")
    
    # Look for patterns in high-mileage multi-day trips
    multiday_cases = [c for c in cases if c['input']['trip_duration_days'] > 1]
    high_efficiency = []
    
    for case in multiday_cases:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        
        mpd = m / d
        if mpd > 100:  # High efficiency trips
            high_efficiency.append((d, m, r, o, mpd))
    
    # Sort by MPD
    high_efficiency.sort(key=lambda x: x[4], reverse=True)
    
    print("Top high-efficiency multi-day trips:")
    for d, m, r, o, mpd in high_efficiency[:10]:
        per_day = o / d
        print(f"  {d}d, {m}mi, ${r:.2f} -> ${o:.2f} ({mpd:.1f} MPD, ${per_day:.2f}/day)")
    
    print("\n=== RECEIPT IMPACT ON MULTI-DAY ===")
    # Find multi-day cases with similar days/miles but different receipts
    
    # Look at 7-8 day cases with 800-1100 miles
    target_cases = []
    for case in cases:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        
        if 7 <= d <= 8 and 800 <= m <= 1100:
            target_cases.append((d, m, r, o))
    
    # Sort by receipt amount
    target_cases.sort(key=lambda x: x[2])
    
    print("7-8 day, 800-1100 mile trips by receipt amount:")
    for d, m, r, o in target_cases[:10]:
        print(f"  {d}d, {m}mi, ${r:.2f} -> ${o:.2f}")

if __name__ == "__main__":
    analyze_multiday_failures() 