# ACME Reimbursement Engine - Reverse Engineering Log

## 🎯 Mission Statement
Reverse engineer a 60-year-old black box reimbursement system by analyzing:
- 1,000 public test cases (input → expected output)
- Employee interviews with anecdotal evidence
- Partial legacy PRD documentation

**Goal**: Build a `run.sh` script that matches ≥99% of public cases with minimal error.

---

## 📊 Key Findings from Data Analysis

### Per-Day Rate Patterns (Critical Discovery)
```
1 day trips:  avg $873/day  (range: $117-$1475)
2 day trips:  avg $523/day  (range: $102-$833)
3 day trips:  avg $337/day  (range: $101-$529)
5 day trips:  avg $255/day  (range: $81-$362)
10 day trips: avg $150/day  (range: $77-$201)
```

**🔑 KEY INSIGHT**: The system doesn't use a fixed daily rate. Instead, shorter trips get massive per-day bonuses.

### Receipt Processing Patterns
- **Low receipts ($0-50)**: Get 2-30x multipliers
- **Medium receipts ($100-200)**: Get ~1.8-2x multipliers  
- **High receipts ($500-600)**: Get ~1x rate
- **Very high receipts ($1000+)**: Get 0.3-0.8x rate (heavily penalized)

### Mileage Component
- Appears to be ~$0.3-0.4 per mile in most cases
- No clear tiered structure like initially assumed

---

## 🧪 Implementation Iterations

### Iteration 1: Initial Hypothesis (FAILED)
**Score**: 33,071 (Average error: $329.71)

**Formula**:
```pseudocode
base = $110 * days
mileage = tiered_rates(1.24, 0.37, 0.29 per mile)
receipts = 0.85 * min(receipts, 500) + 0.25 * excess
efficiency_bonus = 15% if 150-220 MPD
windfall = +$3 if receipts end in .49/.99
```

**Problems**:
- Base rate too high ($110/day vs actual ~$60-150/day depending on length)
- Mileage rates too high
- Didn't account for trip length scaling effect

### Iteration 2: Trip Length Scaling (FAILED)
**Score**: 35,036 (Average error: $349.36)

**Formula**:
```pseudocode
base = variable by trip length (45-60 * days)
mileage = $0.32 per mile
receipts = tiered multipliers (2.5x, 1.8x, 1.0x, 0.3x)
length_adjustment = massive bonuses for 1-2 day trips
windfall = +$3 if receipts end in .49/.99
```

**Problems**:
- 1-day trips still massively over-estimated
- Length adjustment formula wrong (4x multiplier too aggressive)
- High-mileage 1-day trips are the major error source

### Iteration 3: Separate 1-Day vs Multi-Day Logic (MAJOR IMPROVEMENT)
**Score**: 20,751 (Average error: $206.51) - **40% improvement!**

**Formula**:
```pseudocode
if 1-day:
  base = 120
  mileage = min(miles * 0.25, 150)  // Heavy cap
  receipts = complex tiered (0.3x to 1.2x with penalties)
else:
  base = days * (70-40 based on length)
  mileage = miles * 0.3  // No cap
  receipts = simpler tiered (1.0x to 0.8x)
windfall = +$3 if receipts end in .49/.99
```

**Progress**:
- ✅ 1-day trips much more accurate
- ✅ Error cases shifted from 1-day to multi-day trips
- ❌ High-mileage multi-day trips now under-estimated
- ❌ High-receipt multi-day trips over-estimated

### Iteration 4: Aggressive Efficiency Bonuses (MAJOR REGRESSION)
**Score**: 37,174 (Average error: $370.74) - **79% worse than v3!**

**Formula Changes**:
```pseudocode
// Added efficiency bonuses:
if mpd >= 140: mileage = miles * 1.1  // EXTREME bonus
if mpd >= 100: mileage = miles * 0.7
if mpd >= 50:  mileage = miles * 0.4

// More extreme receipt penalties:  
receipts > 1500: only 0.02x rate for excess
```

**CRITICAL FAILURES**:
- 3-day, 1300+ mile trips: Expected $787, Got $1945 (massive over-estimation)
- The 1.1x mileage multiplier is way too high
- Short trips with very high mileage are getting unrealistic bonuses

**Key Learning**: Efficiency bonuses exist but are much more modest than analyzed

---

## 🚨 CRITICAL INSIGHT: Alternative Analysis Script

Found [calculate_reimbursement.py](mdc:analysis-scripts/calculate_reimbursement.py) with **contradictory but valuable insights**:

### Key Revelations:
1. **"Miles/day has NEGATIVE correlation with output"** - Our efficiency bonuses were backwards!
2. **"Receipts dominate (0.704 correlation)"** - Receipt processing is THE primary component
3. **Efficiency PENALTIES not bonuses** for >300 MPD

### Their Formula Approach:
```pseudocode
base = days * 100  // Simple linear base
mileage = tiered_rates(0.58, 0.45, 0.35, 0.25 per mile)
receipts = complex_tiered(1.5x, 1.2x, 0.9x, 0.7x, 0.5x, 0.3x, 0.15x)

// KEY DIFFERENCES:
if mpd > 300: total -= efficiency_penalty  // PENALTY not bonus!
if receipts > 1500: mileage *= reduction_factor  // Receipts suppress mileage
if 1-day && (miles > 800 || receipts > 1500): cap_at(1400-1500)
```

### Critical Corrections Needed:
- ❌ **Remove efficiency bonuses** - they cause negative correlation
- ✅ **Focus on receipt tiering** - it's the dominant factor  
- ✅ **Add efficiency penalties** for unrealistic MPD
- ✅ **Implement receipt-mileage interaction** - high receipts reduce mileage impact

---

## 🔍 Current Error Analysis

### Worst Performing Cases:
```
Case 996: 1d, 1082mi, $1809.49 → Expected: $446.94, Got: $3297.05
Case 899: 1d, 1092mi, $390.55 → Expected: $589.11, Got: $2737.75
Case 421: 1d, 1060mi, $501.67 → Expected: $658.14, Got: $2797.67
```

**Pattern**: All high-error cases are 1-day trips with high mileage. The system seems to cap or heavily penalize these scenarios.

---

## 🧠 Strategic Hypotheses to Test

### Hypothesis A: Base Rate is Fixed Amount, Not Per-Day
```pseudocode
// Instead of: base = rate_per_day * days
// Try: base = fixed_amount / scaling_factor(days)

if days == 1: base = 120
if days == 2: base = 200  
if days == 3: base = 250
if days >= 5: base = 300
```

### Hypothesis B: Mileage Has Daily Limits/Caps
```pseudocode
// High mileage 1-day trips suggest daily mileage caps
daily_miles = miles / days
if daily_miles > 500: apply_penalty()
if daily_miles > 1000: heavy_penalty()
```

### Hypothesis C: Multi-Component Formula with Caps
```pseudocode
per_diem_component = base_by_days[days]
mileage_component = min(miles * rate, daily_cap * days)
receipt_component = diminishing_returns(receipts)
total = per_diem + mileage + receipts + adjustments
```

---

## 📋 UPDATED Strategic Steps (Based on New Insights)

### Phase 1: Receipt-Centric Rebuild  
1. **Implement sophisticated receipt tiering** from alternative analysis
2. **Test $100/day linear base** vs our complex trip-length scaling
3. **Remove ALL efficiency bonuses** - replace with penalties

### Phase 2: Mileage-Receipt Interaction
1. **Implement tiered mileage rates** (0.58 → 0.45 → 0.35 → 0.25)
2. **Add receipt suppression of mileage** for high receipt amounts
3. **Test efficiency penalties** for >300 MPD cases

### Phase 3: Caps and Constraints  
1. **Add 1-day trip caps** (1400-1500 max for high miles/receipts)
2. **Test simple 5-day bonus** (+$20)
3. **Validate windfall rounding** (.49/.99 cases)

### Phase 4: Integration Testing
1. **Compare against alternative script performance**
2. **A/B test key differences** (linear base vs complex, penalties vs bonuses)
3. **Fine-tune based on correlation insights**

---

## 🎛️ Parameters to Experiment With

Based on analysis, focus tuning on:

1. **Base rates by trip length**:
   - 1-day: ~$400-600 total (not per-day)
   - 2-day: ~$400-800 total  
   - 3-day: ~$600-1000 total
   - 5+ day: ~$200-300 per day

2. **Mileage rate**: $0.20-0.40 per mile with potential caps

3. **Receipt multipliers**: 
   - <$50: 1.5-3x
   - $50-200: 1.2-1.8x
   - $200-600: 0.8-1.2x
   - >$600: 0.3-0.8x

---

## 💡 Key Insights for Next Iteration

1. **Stop thinking "per-day"** - the system likely uses trip-total logic
2. **1-day trips are special** - they don't follow normal scaling rules
3. **High mileage gets capped** - especially for short trips
4. **Receipt processing is sophisticated** - strong diminishing returns curve

---

## 🚫 Things That DON'T Work

- Simple per-day multiplication ($X * days)
- Linear mileage rates without caps
- Linear receipt processing
- Efficiency bonuses based on miles-per-day ratios
- Aggressive length adjustment multipliers (4x, 1.5x)

---

---

## 🔍 NEW ANALYSIS: 1-Day Trip Deep Dive

### Critical Discoveries:

1. **High mileage 1-day trips get HEAVILY CAPPED**:
   - 500+ mile trips: Expected $355-678, Simple formula gives $900-2400
   - Clear evidence of mileage caps for single-day trips

2. **Low mileage 1-day trips follow different logic**:
   - Base seems to be ~$120-130 for low-mileage cases
   - Receipt processing varies wildly (rates from -2x to +30x)

3. **Receipt processing is extremely complex**:
   - High receipt amounts ($1000+) get 0.4-0.7x rates
   - Very low receipts get negative components (penalties)
   - Medium receipts get 1-3x multipliers

### New Formula Hypothesis:
```pseudocode
For 1-day trips:
  base = 120
  mileage = min(miles * 0.3, 150)  // Heavy cap at $150
  
  if receipts < 20: receipt_comp = receipts * 0.5  // penalty
  elif receipts < 100: receipt_comp = receipts * 1.5
  elif receipts < 500: receipt_comp = receipts * 0.8
  else: receipt_comp = 400 + (receipts - 500) * 0.3
```

---

## 📈 Success Metrics

- **Target**: <5% average error rate
- **Current**: ~35% average error rate  
- **1-day trips average error**: $215 with new hypothesis (better but still high)
- **Focus**: Refine 1-day trip formula, then tackle multi-day trips
- **Secondary**: Improve medium/long trip accuracy 