#!/usr/bin/env python3
"""Fast evaluation script for reimbursement model"""

import json
import time
import numpy as np
from calculate_reimbursement import calculate_reimbursement


def fast_evaluate(json_file='public_cases.json'):
    """Fast evaluation that processes all cases in batch"""
    
    print("🧾 Black Box Challenge - Fast Evaluation")
    print("=" * 60)
    
    # Load test cases
    start_time = time.time()
    with open(json_file, 'r') as f:
        cases = json.load(f)
    
    print(f"📊 Running evaluation against {len(cases)} test cases...")
    
    # Process all cases
    results = []
    errors = []
    
    for i, case in enumerate(cases):
        if i % 100 == 0 and i > 0:
            print(f"Progress: {i}/{len(cases)} cases processed...")
        
        try:
            # Extract inputs
            trip_days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            # Calculate prediction
            predicted = calculate_reimbursement(trip_days, miles, receipts)
            
            # Calculate error
            error = abs(predicted - expected)
            
            results.append({
                'case_num': i + 1,
                'trip_days': trip_days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': predicted,
                'error': error
            })
            
        except Exception as e:
            errors.append(f"Case {i+1}: {str(e)}")
    
    # Calculate statistics
    if results:
        errors_list = [r['error'] for r in results]
        exact_matches = sum(1 for e in errors_list if e < 0.01)
        close_matches = sum(1 for e in errors_list if e < 1.0)
        avg_error = np.mean(errors_list)
        max_error = max(errors_list)
        max_error_idx = errors_list.index(max_error)
        
        # Display results
        elapsed = time.time() - start_time
        print(f"\n✅ Evaluation Complete in {elapsed:.2f} seconds!")
        print("\n📈 Results Summary:")
        print(f"  Total test cases: {len(cases)}")
        print(f"  Successful runs: {len(results)}")
        print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/len(results)*100:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches} ({close_matches/len(results)*100:.1f}%)")
        print(f"  Average error: ${avg_error:.2f}")
        print(f"  Maximum error: ${max_error:.2f}")
        
        # Calculate score
        score = avg_error * 100 + (len(cases) - exact_matches) * 0.1
        print(f"\n🎯 Your Score: {score:.2f} (lower is better)")
        
        # Show worst cases
        print("\n💡 Highest error cases:")
        sorted_results = sorted(results, key=lambda x: x['error'], reverse=True)
        for r in sorted_results[:5]:
            print(f"  Case {r['case_num']}: {r['trip_days']} days, {r['miles']} miles, ${r['receipts']:.2f} receipts")
            print(f"    Expected: ${r['expected']:.2f}, Got: ${r['predicted']:.2f}, Error: ${r['error']:.2f}")
        
        # Check for patterns in errors
        print("\n📊 Error Analysis:")
        
        # Receipt ending analysis
        receipt_49 = [r for r in results if int(r['receipts'] * 100) % 100 == 49]
        receipt_99 = [r for r in results if int(r['receipts'] * 100) % 100 == 99]
        
        if receipt_49:
            avg_49 = np.mean([r['error'] for r in receipt_49])
            print(f"  .49 receipts: {len(receipt_49)} cases, avg error ${avg_49:.2f}")
        
        if receipt_99:
            avg_99 = np.mean([r['error'] for r in receipt_99])
            print(f"  .99 receipts: {len(receipt_99)} cases, avg error ${avg_99:.2f}")
        
        # Trip length analysis
        for days in [1, 7, 9, 11]:
            day_cases = [r for r in results if r['trip_days'] == days]
            if day_cases:
                avg_day = np.mean([r['error'] for r in day_cases])
                print(f"  {days}-day trips: {len(day_cases)} cases, avg error ${avg_day:.2f}")
    
    else:
        print("❌ No successful test cases!")
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors encountered")
        for e in errors[:5]:
            print(f"  {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        fast_evaluate(sys.argv[1])
    else:
        fast_evaluate() 