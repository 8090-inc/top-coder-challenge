# Review: Cents Hash Discovery v2

## Overview
The script attempts to discover a deterministic hash function that maps trip characteristics to the observed cents endings in the legacy reimbursement system. It uses modular arithmetic (mod 64) to find a perfect hash.

## Strengths

### 1. Mathematical Elegance
- Uses integer arithmetic with mod 64, avoiding floating-point issues
- Perfect hash requirement ensures unique mapping
- Compact representation (64-entry lookup table)

### 2. Performance Optimization
- Early rejection dramatically speeds up search
- Vectorized NumPy operations
- Claims < 0.2s on 1000 rows

### 3. Two-Tier Approach
- First tries simple 3-term hash: `(a*days + b*miles + c*receipts_dollars + d) & 63`
- Falls back to 4-term with receipt cents if needed
- Pragmatic progression from simple to complex

## Potential Issues

### 1. Search Space Limitations
```python
# Basic search: coefficients in [0, 64)
# With cents: coefficients in [0, 32)
```
- Might be too restrictive if true hash uses larger coefficients
- Consider expanding if no perfect hash found

### 2. Feature Engineering
Current features might be insufficient:
- Doesn't account for cluster assignment (our 9 clusters)
- Ignores receipt ending penalties (.49/.99 multipliers)
- No derived features (efficiency, per-day rates)

### 3. Data Assumptions
```python
R = df["receipts"].round().astype(int).to_numpy(np.int16) & 63
```
- Rounds receipts to whole dollars before mod operation
- Might lose important precision

## Recommendations

### 1. Expand Feature Set
Consider additional hash inputs:
```python
# Cluster assignment (0-8)
cluster = get_cluster(days, miles, receipts) & 63

# Receipt ending flag
has_penalty = (receipts % 1 == 0.49) | (receipts % 1 == 0.99)

# Efficiency metric
efficiency = (miles / days) if days > 0 else 0
```

### 2. Try Different Moduli
Instead of fixed mod 64:
```python
for modulo in [64, 100, 128, 256]:
    # Search with different moduli
```

### 3. Consider Non-Linear Terms
```python
# Interaction terms
(a*days*miles + b*receipts + c) % modulo

# Polynomial terms  
(a*days + b*miles^2 + c*receipts) % modulo
```

### 4. Validation Enhancement
Add more detailed validation:
```python
# Check if lookup table matches known patterns
known_cents = {12, 24, 72, 94, ...}  # From hypothesis doc
coverage = sum(1 for v in lut.values() if v in known_cents)
print(f"LUT covers {coverage}/{len(known_cents)} known cents patterns")
```

### 5. Handle Edge Cases
```python
# Special handling for outliers
if is_special_profile(days, miles, receipts):
    # Kevin's special profile, etc.
    return special_cents_lookup(...)
```

## Alternative Approaches

### 1. Cryptographic Hash
```python
# Use first 6 bits of MD5/SHA hash
import hashlib
h = hashlib.md5(f"{days},{miles},{receipts}".encode()).digest()
key = h[0] & 63
```

### 2. CRC-Based
```python
# Use CRC polynomial
import zlib
data = struct.pack('HHH', days, miles, int(receipts))
key = zlib.crc32(data) & 63
```

### 3. Feature Hashing
```python
# Hash multiple features independently
h1 = (days * 17) & 63
h2 = (miles * 31) & 63  
h3 = (int(receipts) * 13) & 63
key = (h1 ^ h2 ^ h3) & 63
```

## Testing Strategy

1. **Known Patterns Test**
   - Verify hash produces cents in {12, 24, 72, 94, ...}
   - Check distribution matches observed frequencies

2. **Collision Test**
   - Ensure no two different inputs map to conflicting cents

3. **Cluster Consistency**
   - Verify hash works across all 9 clusters
   - Check special cases (Kevin's profile, case 86)

4. **Integration Test**
   - Add hash to v5 model
   - Measure impact on exact match rate

## Conclusion

The v2 approach is solid but may need enhancement:
1. Expand search space if initial search fails
2. Consider cluster-aware features
3. Add validation for known cents patterns
4. Test alternative hash functions if simple linear fails

The modular arithmetic approach is clever and efficient. With minor enhancements, it could crack the remaining precision mystery. 