#!/usr/bin/env python3

# Analyze the failing cases from eval.sh
failing_cases = [
    (14, 481, 939.99, 877.17),  # Case 520
    (8, 795, 1645.99, 644.69),  # Case 684 
    (14, 296, 485.68, 924.90),  # Case 389
    (11, 740, 1171.99, 902.09), # Case 367
    (14, 68, 438.96, 866.76)    # Case 633
]

print("=== HIGH-ERROR CASES ANALYSIS ===")
print("Days\tMiles\tReceipts\tExpected\t$/Day\tMiles/Day")
print("-" * 65)

for d, m, r, expected in failing_cases:
    per_day = expected / d
    miles_per_day = m / d
    print(f"{d}\t{m}\t${r:.2f}\t\t${expected:.2f}\t\t${per_day:.2f}\t{miles_per_day:.1f}")

print("\n=== OBSERVATIONS ===")
print("• All expected outputs are much lower than our current formula produces")
print("• Per-day rates are reasonable: $58-80 range")  
print("• High receipt cases (like $1645) don't get proportional reimbursement")
print("• Long trips (14 days) seem to have consistent per-day rates around $60-66")

# Let's estimate what the base formula might actually be
print("\n=== REVERSE ENGINEERING ATTEMPT ===")
for d, m, r, expected in failing_cases:
    # Try different base rates
    base_50 = d * 50
    base_60 = d * 60
    remaining_after_50 = expected - base_50
    remaining_after_60 = expected - base_60
    
    print(f"\nCase: {d}d, {m}mi, ${r:.2f} -> ${expected:.2f}")
    print(f"  If $50/day base: ${base_50:.2f}, remaining: ${remaining_after_50:.2f}")
    print(f"  If $60/day base: ${base_60:.2f}, remaining: ${remaining_after_60:.2f}")
    
    # Estimate mileage component
    if remaining_after_60 > 0:
        implied_mileage_rate = remaining_after_60 / m
        print(f"  Implied mileage rate: ${implied_mileage_rate:.3f}/mile") 