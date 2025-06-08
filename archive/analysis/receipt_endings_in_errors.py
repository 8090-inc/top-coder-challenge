import pandas as pd
import numpy as np

# Load v5 predictions
df = pd.read_csv('../public_cases_predictions_v5.csv')
df['error'] = np.abs(df['predicted'] - df['expected_output'])
df['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100

print("RECEIPT ENDINGS ANALYSIS IN V5 ERRORS")
print("=" * 60)

# Overall distribution
print("\nOverall receipt endings distribution:")
overall_dist = df['receipt_cents'].value_counts().head(20)
print(overall_dist)

# Distribution in high-error cases
high_error = df[df['error'] > 100]
print(f"\n\nReceipt endings in high-error cases (error > $100):")
high_error_dist = high_error['receipt_cents'].value_counts().head(20)
print(high_error_dist)

# Check specific problematic endings
print("\n\nAnalysis of .49 and .99 endings:")
print("-" * 60)

# .49 endings
ending_49 = df[df['receipt_cents'] == 49]
print(f"\nReceipts ending in .49:")
print(f"  Count: {len(ending_49)}")
print(f"  Average error: ${ending_49['error'].mean():.2f}")
print(f"  Average expected: ${ending_49['expected_output'].mean():.2f}")
print(f"  Average predicted: ${ending_49['predicted'].mean():.2f}")

# .99 endings
ending_99 = df[df['receipt_cents'] == 99]
print(f"\nReceipts ending in .99:")
print(f"  Count: {len(ending_99)}")
print(f"  Average error: ${ending_99['error'].mean():.2f}")
print(f"  Average expected: ${ending_99['expected_output'].mean():.2f}")
print(f"  Average predicted: ${ending_99['predicted'].mean():.2f}")

# Other endings for comparison
other_endings = df[~df['receipt_cents'].isin([49, 99])]
print(f"\nOther receipt endings:")
print(f"  Count: {len(other_endings)}")
print(f"  Average error: ${other_endings['error'].mean():.2f}")
print(f"  Average expected: ${other_endings['expected_output'].mean():.2f}")
print(f"  Average predicted: ${other_endings['predicted'].mean():.2f}")

# Check if v5 is already handling the penalty
print("\n\nChecking if v5 handles receipt penalties:")
print("-" * 60)

# For .49 endings, expected should be ~34.1% of what it would be without penalty
# For .99 endings, expected should be ~51% of what it would be without penalty

for idx, row in ending_49.head(5).iterrows():
    print(f"\n.49 case: {row['trip_days']:.0f} days, {row['miles']:.0f} miles, ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected_output']:.2f}")
    print(f"  Predicted: ${row['predicted']:.2f}")
    print(f"  Error: ${row['error']:.2f} ({row['error']/row['expected_output']*100:.1f}%)")
    # Estimate what it would be without penalty
    estimated_no_penalty = row['expected_output'] / 0.341
    print(f"  Estimated without penalty: ${estimated_no_penalty:.2f}")

print("\n" + "-" * 40)

for idx, row in ending_99.head(5).iterrows():
    print(f"\n.99 case: {row['trip_days']:.0f} days, {row['miles']:.0f} miles, ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected_output']:.2f}")
    print(f"  Predicted: ${row['predicted']:.2f}")
    print(f"  Error: ${row['error']:.2f} ({row['error']/row['expected_output']*100:.1f}%)")
    # Estimate what it would be without penalty
    estimated_no_penalty = row['expected_output'] / 0.51
    print(f"  Estimated without penalty: ${estimated_no_penalty:.2f}") 