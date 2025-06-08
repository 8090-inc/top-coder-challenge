# Interview Claims vs Hypothesis Testing Results

## Executive Summary
The interviews contain a mix of accurate insights, partially true observations, and false information. Most importantly, the interviews correctly identified that there are multiple calculation paths (Kevin's "6 paths" became our 9 clusters) and that context matters more than universal rules.

## Detailed Cross-Reference Analysis

### 1. BASE CALCULATION CLAIMS

| Claim | Source | Status | Reality |
|-------|--------|--------|---------|
| "$100/day base per diem" | Lisa | 🔴 MISLEADING | Actual: $0-$182.45 base + $0-$74.81/day depending on cluster |
| "Mileage rate drops after 100 miles" | Lisa | 🟡 PARTIAL | Mileage importance varies by cluster, can even be negative |
| "58 cents per mile base rate" | Lisa | 🔴 FALSE | Actual: $0.138-$0.434/mile (cluster-dependent) |
| "Receipt caps exist" | Lisa | 🟢 TRUE | Cluster 0 has $1800 cap; coverage ratios decrease |

### 2. TRIP DURATION EFFECTS

| Claim | Source | Status | Reality |
|-------|--------|--------|---------|
| "5-day trips get bonus" | Lisa, Marcus | 🔴 FALSE | No universal 5-day bonus found |
| "Sweet spot 4-6 days" | Jennifer | 🟡 CONTEXT | True for some clusters, not universal |
| "8+ days penalized" | Kevin | 🟡 CONTEXT | Depends on cluster - Cluster 2 handles long trips well |

### 3. EFFICIENCY CLAIMS

| Claim | Source | Status | Reality |
|-------|--------|--------|---------|
| "180-220 mi/day optimal" | Kevin | 🔴 MISLEADING | That range actually shows LOWER reimbursement |
| "400+ mi/day penalized" | Kevin | 🟡 PARTIAL | True for some clusters but not all |
| "System rewards hustle" | Marcus | 🟢 CONCEPTUAL | Different clusters reward different behaviors |

### 4. RECEIPT PATTERNS

| Claim | Source | Status | Reality |
|-------|--------|--------|---------|
| ".49/.99 endings get bonus" | Lisa | 🟢 REVERSED | These endings get PENALTIES (-65.9% and -49%) |
| "Low receipts penalized" | Dave, Jennifer | 🟢 TRUE | <$50 significantly penalized |
| "$50 threshold" | Lisa | 🟢 CONFIRMED | Major behavioral change at ~$50 |
| "Medium-high best ($600-800)" | Lisa | 🟡 CONTEXT | Depends on cluster assignment |

### 5. CALCULATION PATHS

| Claim | Source | Status | Reality |
|-------|--------|--------|---------|
| "6 different calculation paths" | Kevin | 🟢 CLOSE | Found 9 clusters (expanded from 6) |
| "k-means clustering would work" | Kevin | 🟢 TRUE | Successfully used k-means to find clusters |
| "Different paths for trip types" | Kevin | 🟢 TRUE | Each cluster represents different trip profile |

### 6. TIMING/RANDOMNESS

| Claim | Source | Status | Reality |
|-------|--------|--------|---------|
| "Tuesday 8% better" | Kevin | ⚪ UNTESTABLE | No date data available |
| "Lunar cycles matter" | Kevin | ⚪ UNTESTABLE | No date data available |
| "Q4 more generous" | Marcus | ⚪ UNTESTABLE | No temporal data |
| "System has randomness" | Dave | 🔴 UNLIKELY | Model achieves 89.9% accuracy - mostly deterministic |

### 7. USER/DEPARTMENT EFFECTS

| Claim | Source | Status | Reality |
|-------|--------|--------|---------|
| "Sales does better" | Jennifer | ⚪ UNKNOWN | No department data |
| "System remembers history" | Marcus, Kevin | ⚪ UNKNOWN | No user ID data |
| "Different rules by department" | Marcus | 🟡 POSSIBLE | Could explain some variance |

## Key Insights from Analysis

### What the Interviews Got RIGHT:
1. **Multiple calculation paths exist** - Kevin's intuition about 6 paths was close (we found 9)
2. **Context matters more than universal rules** - Every "rule" is cluster-specific
3. **Receipt endings matter** - But opposite effect (penalty not bonus)
4. **Low receipts are penalized** - Confirmed with strong evidence
5. **Efficiency concepts exist** - But implementation differs from claims

### What the Interviews Got WRONG:
1. **Universal 5-day bonus** - Doesn't exist
2. **Simple per diem rate** - It's cluster-dependent
3. **Linear mileage rates** - Can be negative in some clusters!
4. **180-220 mi/day optimal** - Actually performs worse

### What Remains UNCERTAIN:
1. **Temporal effects** - No date data to test
2. **User/department variations** - No user data
3. **Historical effects** - No user history available

## Strategic Insights

### The interviews were likely designed to:
1. **Encode the cluster concept** - Multiple people mentioned different "paths"
2. **Misdirect on specifics** - Wrong numbers but right concepts
3. **Hide key insights** - Receipt ending penalty presented as bonus
4. **Test reasoning ability** - Mix of true patterns and false specifics

### Most Valuable Interview Insights:
1. **Kevin's clustering approach** - Led directly to solution
2. **Lisa's receipt observations** - Identified critical thresholds
3. **General concept of multiple paths** - Key to cracking the system

## Recommendations for Next Steps

1. **Trust conceptual insights, verify specific numbers**
2. **The 9-cluster model aligns with interview hints about complexity**
3. **Focus on patterns that multiple sources mentioned**
4. **Be skeptical of precise numbers from interviews**
5. **The "randomness" people perceive is likely cluster assignment confusion**

## Conclusion

The interviews were a clever mix of truth and misdirection. They correctly pointed toward a multi-path system with context-dependent rules, but provided incorrect specifics to prevent simple reverse engineering. The key was recognizing the pattern (multiple calculation methods) while ignoring the false precision. 