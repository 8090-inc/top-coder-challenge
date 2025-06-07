import json

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculates the travel reimbursement amount based on the legacy system's logic.
    """
    # Initial baseline: A simple calculation to be refined later
    reimbursement = 0

    # Calculate miles_per_day early as it's used in multiple places
    miles_per_day = 0
    if trip_duration_days > 0:
        miles_per_day = miles_traveled / trip_duration_days

    # Per diem: A common factor mentioned
    per_diem_rate = 100  # Base rate from Lisa's interview
    reimbursement += trip_duration_days * per_diem_rate

    # Bonus for 5-day trips (Lisa's observation)
    five_day_bonus = 0
    if trip_duration_days == 5:
        if miles_per_day < 30:
            five_day_bonus = 10 # Different bonus for this path
        else:
            five_day_bonus = 25
    reimbursement += five_day_bonus

    # Tiered Mileage calculation (Lisa's observation)
    mileage_rate_tier1 = 0.58  # For first 100 miles
    mileage_rate_tier2 = 0.48  # For miles thereafter (tweaked)

    if miles_traveled <= 100:
        reimbursement += miles_traveled * mileage_rate_tier1
    else:
        reimbursement += (100 * mileage_rate_tier1) + ((miles_traveled - 100) * mileage_rate_tier2)

    # Receipt processing logic (based on Lisa's and Dave's comments)
    receipt_reimbursement = 0
    # Penalty for very low receipts (Dave: "$12 receipt, better off submitting nothing")
    # Lisa: "if you submit $50 in receipts for a multi-day trip, you're better off submitting nothing"
    # This threshold might depend on trip_duration_days
    low_receipt_threshold = 0
    if trip_duration_days == 1:
        low_receipt_threshold = 10
    elif trip_duration_days <= 3:
        low_receipt_threshold = 20 # Stricter for short multi-day
    else: # Longer trips might have slightly more lenient low receipt threshold
        low_receipt_threshold = 30

    if total_receipts_amount < low_receipt_threshold:
        # This is a tricky part. "Worse than per diem" or "less than per diem" implies a penalty.
        # For now, let's say it just doesn't add much or could even slightly penalize.
        # A simple start: no reimbursement for very low receipts.
        # A more aggressive penalty could be something like -50 if it's very low.
        # Let's try no reimbursement first.
        receipt_reimbursement = 0
        # If we want to implement "worse than per diem", we might subtract a value if receipts are submitted and low.
        # For example: if 0 < total_receipts_amount < low_receipt_threshold: reimbursement -= 20 (penalty)
    # miles_per_day is already calculated above

    if total_receipts_amount < low_receipt_threshold:
        receipt_reimbursement = 0
    elif trip_duration_days == 5 and miles_per_day < 30: # Special path for low-efficiency 5-day trips
        # Use LESS generous tiers for this specific case
        if total_receipts_amount <= 50:
            receipt_reimbursement = total_receipts_amount * 0.8
        elif total_receipts_amount <= 200: # 50*0.8 = 40
            receipt_reimbursement = 40 + (total_receipts_amount - 50) * 0.6
        elif total_receipts_amount <= 500: # 40 + 150*0.6 = 40 + 90 = 130
            receipt_reimbursement = 130 + (total_receipts_amount - 200) * 0.4
        elif total_receipts_amount <= 1000: # 130 + 300*0.4 = 130 + 120 = 250
            receipt_reimbursement = 250 + (total_receipts_amount - 500) * 0.2
        else: # Amounts over 1000. 250 + 500*0.2 = 250 + 100 = 350
            receipt_reimbursement = 350 + (total_receipts_amount - 1000) * 0.1
    else:
        # Use MORE generous placeholder tiers for other cases (Updated):
        if total_receipts_amount <= 50:
            receipt_reimbursement = total_receipts_amount * 1.0
        elif total_receipts_amount <= 200: # Max at this tier: 50 + 150*0.8 = 170
            receipt_reimbursement = 50 + (total_receipts_amount - 50) * 0.8
        elif total_receipts_amount <= 500: # Max at this tier: 170 + 300*0.6 = 350
            receipt_reimbursement = 170 + (total_receipts_amount - 200) * 0.6
        elif total_receipts_amount <= 1000: # Max at this tier: 170 + 300*0.6 = 350. Then 350 + 500*0.5 = 600
            receipt_reimbursement = 350 + (total_receipts_amount - 500) * 0.5
        else: # Amounts over 1000. Max at this tier: 600 + (R-1000)*0.60
            receipt_reimbursement = 600 + (total_receipts_amount - 1000) * 0.60

    reimbursement += receipt_reimbursement

    # Efficiency Modifier (Kevin's observations)
    # miles_per_day is already calculated above
    efficiency_modifier = 0
    if trip_duration_days > 1: # Penalties typically for multi-day trips
        if miles_per_day < 30: # Extremely low efficiency
            efficiency_modifier = -178  # Fine-tuned large penalty
        elif miles_per_day < 50: # Moderately low efficiency
            efficiency_modifier = -70   # Adjusted moderate penalty

    # Bonuses can apply to any trip length if efficiency is high
    if 150 <= miles_per_day <= 250: # Kevin's "sweet spot"
        efficiency_modifier = 75   # Placeholder bonus
    elif miles_per_day > 250: # Kevin: "Go too high, the bonuses start dropping off again"
        efficiency_modifier = 25   # Reduced bonus for very high efficiency

    reimbursement += efficiency_modifier

    # Lisa: "If your receipts end in 49 or 99 cents, you often get a little extra money."
    cents = round(total_receipts_amount * 100) % 100
    if cents == 49 or cents == 99:
        reimbursement += 2.50 # Arbitrary small bonus for .49 or .99 receipts

    return round(reimbursement, 2)

if __name__ == "__main__":
    # Load public cases
    with open("public_cases.json", "r") as f:
        public_cases = json.load(f)

    # Test the calculation
    print("--- Testing 5-day trips and trips with >100 miles ---")
    five_day_trip_cases_shown = 0
    high_mileage_cases_shown = 0

    for i, case in enumerate(public_cases):
        input_data = case["input"]
        expected_output = case["expected_output"]

        show_case = False
        if input_data["trip_duration_days"] == 5 and five_day_trip_cases_shown < 5:
            show_case = True
            five_day_trip_cases_shown += 1
        elif input_data["miles_traveled"] > 100 and high_mileage_cases_shown < 5 and not (input_data["trip_duration_days"] == 5): # Avoid duplicating 5-day trips if they also have high mileage
            # Let's find some high mileage cases that are NOT 5-day trips to isolate mileage logic
            # We also need to ensure we haven't already shown 5 of these
            # This logic can be complex, so we'll find the first few distinct ones.
            # A simpler way for now is to just iterate and pick the first few that satisfy the mileage condition
            # and haven't been shown for the 5-day condition.
            # To ensure we get 5 distinct high mileage cases not overlapping with 5-day shown cases,
            # we might need a more robust selection, but this should give us a good start.

            # Simplified selection for now:
            # Find up to 5 high mileage cases that are not also 5-day trips (if we still need to show high mileage cases)
            # This is tricky because we might exhaust 5-day cases that are also high mileage.
            # For now, let's just print the first 5 of each category found, even if there's overlap in properties.
            # The goal is to see the logic in action.

            # Re-evaluating the print condition to be simpler:
            # Show if it's a 5-day trip (up to 5 times)
            # OR if it's a high mileage trip (up to 5 times), and not already shown as a 5-day trip.
            # This still might not give 5 *distinct* high mileage if they are all 5-day trips.
            # A better approach: just iterate and print if it's a 5-day trip OR a high mileage trip, up to a total of ~10 diverse cases.
            pass # Will simplify printing logic below.

    # Simpler printing: Show first 5 5-day trips, then first 5 (non-5-day) high-mileage trips.

    cases_to_show = []
    five_day_trips = [c for c in public_cases if c["input"]["trip_duration_days"] == 5]
    high_mileage_trips = [c for c in public_cases if c["input"]["miles_traveled"] > 100 and c["input"]["trip_duration_days"] != 5]

    cases_to_show.extend(five_day_trips[:5])
    # Ensure we don't add duplicates if a trip is both 5-day and high mileage (already handled by second list comprehension)
    # Add high mileage cases, ensuring we have a variety
    needed_high_mileage = 5
    for hm_case in high_mileage_trips:
        if len(cases_to_show) < 10 and hm_case not in cases_to_show: # Max 10 total, avoid duplicates
             cases_to_show.append(hm_case)
             needed_high_mileage -=1
             if needed_high_mileage == 0:
                 break

    # Also add the first few overall cases to see general performance
    overall_cases_to_add = 3
    for case in public_cases:
        if overall_cases_to_add == 0:
            break
        if case not in cases_to_show:
            cases_to_show.append(case)
            overall_cases_to_add -=1

    print(f"--- Displaying {len(cases_to_show)} selected test cases ---")
    for case in cases_to_show:
        input_data = case["input"]
        expected_output = case["expected_output"]
        calculated_output = calculate_reimbursement(
            input_data["trip_duration_days"],
            input_data["miles_traveled"],
            input_data["total_receipts_amount"]
        )
        print(f"Input: {input_data}")
        print(f"Expected: {expected_output}, Calculated: {calculated_output}, Diff: {round(expected_output - calculated_output, 2)}")
        print("-" * 20)
