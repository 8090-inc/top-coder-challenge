#!/usr/bin/env python3
import json
import statistics

def analyze_one_day_trips():
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Focus on 1-day trips
    one_day_cases = [c for c in cases if c['input']['trip_duration_days'] == 1]
    print(f"=== 1-DAY TRIP ANALYSIS ({len(one_day_cases)} cases) ===\n")
    
    # Sort by mileage to see patterns
    one_day_cases.sort(key=lambda x: x['input']['miles_traveled'])
    
    print("1. MILEAGE vs PAYOUT ANALYSIS")
    print("-" * 50)
    print("Miles\tReceipts\tOutput\t\t$/Mile\tReceipt Rate")
    
    for case in one_day_cases[:20]:  # First 20 cases by mileage
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        
        # Rough estimates
        per_mile = o / m if m > 0 else 0
        
        # Estimate receipt component (assume base + mileage, see what's left)
        est_base = 100  # Rough guess
        est_mileage = m * 0.5  # Rough guess
        receipt_component = o - est_base - est_mileage
        receipt_rate = receipt_component / r if r > 0 else 0
        
        print(f"{m}\t${r:.2f}\t\t${o:.2f}\t\t${per_mile:.3f}\t{receipt_rate:.2f}")
    
    print("\n2. HIGH MILEAGE 1-DAY TRIPS (>500 miles)")
    print("-" * 50)
    high_mileage = [c for c in one_day_cases if c['input']['miles_traveled'] > 500]
    
    for case in high_mileage[:10]:
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        
        # What would a simple $0.5/mile + $100 base give?
        simple_estimate = 100 + (m * 0.5) + r
        print(f"{m}mi, ${r:.2f} -> ${o:.2f} (simple est: ${simple_estimate:.2f}, diff: ${o - simple_estimate:.2f})")
    
    print("\n3. LOW MILEAGE 1-DAY TRIPS (<100 miles)")
    print("-" * 50)
    low_mileage = [c for c in one_day_cases if c['input']['miles_traveled'] < 100]
    
    for case in low_mileage[:10]:
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        
        # For low mileage, receipts likely dominate
        est_base_mileage = 100 + (m * 0.5)
        receipt_component = o - est_base_mileage
        print(f"{m}mi, ${r:.2f} -> ${o:.2f} (base+miles: ${est_base_mileage:.2f}, receipt comp: ${receipt_component:.2f})")
    
    print("\n4. RECEIPT IMPACT ANALYSIS")
    print("-" * 50)
    print("Grouping similar mileage, different receipts...")
    
    # Group by mileage ranges
    mileage_groups = {
        "50-100": [c for c in one_day_cases if 50 <= c['input']['miles_traveled'] <= 100],
        "200-300": [c for c in one_day_cases if 200 <= c['input']['miles_traveled'] <= 300],
        "500-600": [c for c in one_day_cases if 500 <= c['input']['miles_traveled'] <= 600],
    }
    
    for group_name, group_cases in mileage_groups.items():
        if len(group_cases) >= 3:
            print(f"\n{group_name} miles:")
            # Sort by receipt amount
            group_cases.sort(key=lambda x: x['input']['total_receipts_amount'])
            for case in group_cases[:5]:
                m = case['input']['miles_traveled']
                r = case['input']['total_receipts_amount']
                o = case['expected_output']
                print(f"  {m}mi, ${r:.2f} -> ${o:.2f}")
    
    print("\n5. FORMULA HYPOTHESIS TESTING")
    print("-" * 50)
    
    # Test Hypothesis: Base = 100, Mileage = min(miles * 0.5, 200), Receipts = complex
    print("Testing: Base=$100 + Mileage=min(miles*0.5, $200) + Receipt_component")
    
    errors = []
    for case in one_day_cases[:20]:
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        o = case['expected_output']
        
        # Test formula
        base = 100
        mileage = min(m * 0.5, 200)  # Cap at $200
        
        # Back-calculate what receipt component should be
        required_receipt_component = o - base - mileage
        implied_rate = required_receipt_component / r if r > 0 else 0
        
        print(f"{m}mi, ${r:.2f} -> ${o:.2f}")
        print(f"  Base: ${base}, Mileage: ${mileage:.2f}, Required receipt: ${required_receipt_component:.2f} (rate: {implied_rate:.2f})")
        
        # Test our hypothesis error
        if r <= 50:
            receipt_comp = r * 2.0
        elif r <= 200:
            receipt_comp = r * 1.5
        else:
            receipt_comp = r * 0.8
            
        estimate = base + mileage + receipt_comp
        error = abs(o - estimate)
        errors.append(error)
        print(f"  Estimate: ${estimate:.2f}, Error: ${error:.2f}")
        print()
    
    avg_error = statistics.mean(errors)
    print(f"Average error with hypothesis: ${avg_error:.2f}")

if __name__ == "__main__":
    analyze_one_day_trips() 