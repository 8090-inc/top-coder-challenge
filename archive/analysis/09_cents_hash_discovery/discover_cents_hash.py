#!/usr/bin/env python3
"""
Discover the cents hash function used by the legacy system.

Strategy:
1. Build evidence table with potential hash inputs
2. Try ExtraTreesClassifier to learn the pattern
3. Brute-force search for linear hash if needed
4. Validate the discovered hash + LUT
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report
import json

# Load data
print("Loading data...")
df = pd.read_csv('../../public_cases_predictions_v4.csv')

# Step 1: Build evidence table
print("\nStep 1: Building evidence table...")
evidence = pd.DataFrame()

# Core features
evidence['c_out'] = (df['expected_output'] * 100).astype(int) % 100  # Expected cents (0-99)
evidence['d'] = df['trip_days'].astype(int)                          # Days
evidence['m'] = df['miles'].round().astype(int)                      # Integer miles
evidence['r100'] = df['receipts'].round().astype(int)                # Whole-dollar receipts
evidence['centsR'] = (df['receipts'] * 100).astype(int) % 100        # Receipt cents
evidence['rule'] = df['predicted'].astype(int)                       # Rule engine dollars

# Derived features that might matter
evidence['m_mod_64'] = evidence['m'] % 64
evidence['m_mod_32'] = evidence['m'] % 32
evidence['d_mod_7'] = evidence['d'] % 7
evidence['r100_mod_64'] = evidence['r100'] % 64

# Low-entropy combinations
evidence['combo1'] = (evidence['d'].values << 2) ^ (evidence['m'].values % 64)
evidence['combo2'] = (evidence['d'].values * 17 + evidence['m'].values) % 64
evidence['combo3'] = (evidence['d'].values + evidence['m'].values + evidence['r100'].values) % 64

# Print unique cents distribution
unique_cents = evidence['c_out'].nunique()
cents_counts = evidence['c_out'].value_counts()
print(f"\nUnique cents values: {unique_cents}")
print(f"Min occurrences: {cents_counts.min()}, Max occurrences: {cents_counts.max()}")
print(f"Top 10 most common cents: {list(cents_counts.head(10).index)}")

# Step 2: Try ExtraTreesClassifier
print("\n" + "="*60)
print("Step 2: Training ExtraTreesClassifier...")

# Features for classification
feature_cols = ['d', 'm', 'r100', 'centsR', 'm_mod_64', 'm_mod_32', 
                'd_mod_7', 'r100_mod_64', 'combo1', 'combo2', 'combo3']
X = evidence[feature_cols]
y = evidence['c_out']

# Train classifier
clf = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

clf.fit(X, y)
y_pred = clf.predict(X)
accuracy = (y_pred == y).mean()

print(f"\nClassifier accuracy: {accuracy:.1%}")

# Feature importance
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop feature importances:")
print(importances.head())

if accuracy < 1.0:
    # Analyze misclassifications
    misses = evidence[y_pred != y]
    print(f"\nMisclassified cases: {len(misses)}")
    print("\nConfusion patterns (showing top mismatches):")
    for i, (idx, row) in enumerate(misses.head(10).iterrows()):
        print(f"  Case {idx}: Expected {row['c_out']:02d}, Predicted {y_pred[idx]:02d}")

# Step 3: Brute-force search for linear hash
print("\n" + "="*60)
print("Step 3: Brute-force searching for linear hash mod 64...")

# Try to find a hash function that maps to unique cents
best_hashes = []

# Prepare arrays for vectorized computation
D = evidence['d'].values
M = evidence['m'].values
R = evidence['r100'].values
C = evidence['c_out'].values
CR = evidence['centsR'].values

# Try different combinations
print("\nSearching for perfect hash (this may take a moment)...")

# First try without receipt cents
found_perfect = False
for a in range(64):
    if a % 16 == 0:
        print(f"  Progress: {a}/64...")
    for b in range(64):
        for c in range(64):
            for d0 in range(64):
                # Compute hash
                key = ((a * D + b * M + c * R + d0) & 63)
                
                # Check if each key maps to unique cents
                key_df = pd.DataFrame({'key': key, 'cents': C})
                grouped = key_df.groupby('key')['cents'].nunique()
                
                if grouped.max() == 1:
                    # Perfect hash found!
                    print(f"\n🎉 FOUND PERFECT HASH: ({a}*days + {b}*miles + {c}*receipts + {d0}) % 64")
                    best_hashes.append((a, b, c, d0, 'basic'))
                    found_perfect = True
                    break
            if found_perfect:
                break
        if found_perfect:
            break
    if found_perfect:
        break

# If not found, try including receipt cents
if not found_perfect:
    print("\nNo perfect hash found with basic inputs. Trying with receipt cents...")
    for a in range(32):
        if a % 8 == 0:
            print(f"  Progress: {a}/32...")
        for b in range(32):
            for c in range(32):
                for e in range(32):  # coefficient for receipt cents
                    # Compute hash including receipt cents
                    key = ((a * D + b * M + c * R + e * CR) & 63)
                    
                    # Check if each key maps to unique cents
                    key_df = pd.DataFrame({'key': key, 'cents': C})
                    grouped = key_df.groupby('key')['cents'].nunique()
                    
                    if grouped.max() == 1:
                        # Perfect hash found!
                        print(f"\n🎉 FOUND PERFECT HASH: ({a}*days + {b}*miles + {c}*receipts + {e}*receipt_cents) % 64")
                        best_hashes.append((a, b, c, e, 'with_cents'))
                        found_perfect = True
                        break
                if found_perfect:
                    break
            if found_perfect:
                break
        if found_perfect:
            break

# Build lookup table for the best hash
if best_hashes:
    # Use the first perfect hash found
    if best_hashes[0][4] == 'basic':
        a, b, c, d0, _ = best_hashes[0]
        key = ((a * D + b * M + c * R + d0) & 63)
    else:
        a, b, c, e, _ = best_hashes[0]
        key = ((a * D + b * M + c * R + e * CR) & 63)
    
    # Build lookup table
    key_df = pd.DataFrame({'key': key, 'cents': C})
    lut = key_df.groupby('key')['cents'].first().sort_index()
    
    print(f"\nLookup table (64 entries):")
    print("Key -> Cents")
    for k in range(64):
        if k in lut:
            print(f"{k:3d} -> {lut[k]:02d}", end="    ")
            if (k + 1) % 8 == 0:
                print()
    
    # Save the hash function and LUT
    hash_config = {
        'type': best_hashes[0][4],
        'coefficients': {
            'days': int(a),
            'miles': int(b),
            'receipts_dollars': int(c),
            'receipts_cents': int(e) if best_hashes[0][4] == 'with_cents' else 0,
            'offset': int(d0) if best_hashes[0][4] == 'basic' else 0
        },
        'lookup_table': {int(k): int(v) for k, v in lut.items()}
    }
    
    with open('cents_hash_config.json', 'w') as f:
        json.dump(hash_config, f, indent=2)
    
    print(f"\n✅ Hash configuration saved to cents_hash_config.json")
    
    # Validate on training data
    print("\nValidating on training data...")
    errors = 0
    for i in range(len(evidence)):
        if best_hashes[0][4] == 'basic':
            k = ((a * D[i] + b * M[i] + c * R[i] + d0) & 63)
        else:
            k = ((a * D[i] + b * M[i] + c * R[i] + e * CR[i]) & 63)
        
        predicted_cents = lut[k]
        if predicted_cents != C[i]:
            errors += 1
            if errors <= 5:
                print(f"  Error: Case {i}, Key {k}, Expected {C[i]}, Got {predicted_cents}")
    
    print(f"\nValidation errors: {errors}/{len(evidence)} ({errors/len(evidence)*100:.1%})")
    
else:
    print("\n❌ No perfect hash found. Need to explore more complex functions.")
    
    # Analyze why we couldn't find a perfect hash
    print("\nAnalyzing cents distribution for patterns...")
    
    # Check if certain combinations always map to same cents
    for col in ['d', 'm_mod_64', 'r100_mod_64']:
        print(f"\nGrouping by {col}:")
        grouped = evidence.groupby(col)['c_out'].apply(lambda x: len(x.unique()))
        print(f"  Groups with unique cents mapping: {(grouped == 1).sum()}")
        print(f"  Max different cents per group: {grouped.max()}")

print("\n" + "="*60)
print("Analysis complete!") 