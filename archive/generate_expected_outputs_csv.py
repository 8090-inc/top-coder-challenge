import json
import pandas as pd

# Load public cases
with open('data/raw/public_cases.json', 'r') as f:
    cases = json.load(f)

# Extract data - using actual expected outputs
data = []
for case in cases:
    data.append({
        'trip_days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'],
        'receipts': case['input']['total_receipts_amount'],
        'expected_output': case['expected_output']  # This is the actual legacy system output
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save for cents hash discovery
df.to_csv('public_cases_expected_outputs.csv', index=False)

# Analyze cents patterns in actual data
cents = ((df['expected_output'] * 100).round().astype(int) % 100)
unique_cents = sorted(cents.unique())

print(f"Generated CSV with {len(df)} cases")
print(f"\nCents analysis of ACTUAL expected outputs:")
print(f"Unique cents values: {len(unique_cents)}")
print(f"Top 10 most common: {cents.value_counts().head(10).index.tolist()}")
print(f"Example cents: {unique_cents[:20]}")

# Check if they match the known pattern
known_pattern_cents = {12, 24, 72, 94}  # From hypothesis doc
matching = sum(1 for c in unique_cents if c in known_pattern_cents)
print(f"\nMatching known pattern cents: {matching}/{len(known_pattern_cents)}") 