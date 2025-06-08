"""
Practical Ensemble Model v6 - with improved Cluster 0 handling

This model improves on v5 by incorporating the Cluster 0 improvements
that showed 6.2% improvement for that cluster and 3.7% overall improvement
to the rule engine.
"""

import pandas as pd
import numpy as np
from cluster_models_optimized_v6 import calculate_reimbursement_v6
from cluster_models_optimized import calculate_reimbursement_v3


def get_receipt_confidence(receipts):
    """Categorize receipt confidence levels"""
    if receipts == 0:
        return 'zero'
    elif receipts < 50:
        return 'very_low'
    elif receipts < 200:
        return 'low'  
    elif receipts < 1000:
        return 'medium'
    elif receipts < 2000:
        return 'high'
    else:
        return 'very_high'


def get_trip_complexity_features(trip_days, miles, receipts):
    """Extract trip complexity indicators"""
    miles_per_day = miles / trip_days if trip_days > 0 else 0
    receipts_per_day = receipts / trip_days if trip_days > 0 else 0
    
    return {
        'standard_trip': int(
            2 <= trip_days <= 10 and
            50 <= miles_per_day <= 300 and
            50 <= receipts_per_day <= 400
        ),
        'extreme_case': int(
            trip_days > 20 or
            miles_per_day > 500 or
            receipts_per_day > 800 or
            (trip_days == 1 and miles > 800)
        ),
        'miles_per_day': miles_per_day,
        'receipts_per_day': receipts_per_day
    }


def calculate_reimbursement_practical_ensemble_v6(trip_days, miles, receipts):
    """
    Version 6 of the practical ensemble using:
    - Improved rule engine (v6) with Cluster 0 optimization
    - Simple ML-based adjustments for specific patterns
    - Conservative corrections to avoid overfitting
    
    This version focuses on the rule engine improvements rather than
    complex ML models.
    """
    
    # Get base prediction from improved rule engine
    v6_pred = calculate_reimbursement_v6(trip_days, miles, receipts)
    
    # Get v3 prediction for comparison
    v3_pred = calculate_reimbursement_v3(trip_days, miles, receipts)
    
    # Start with v6 prediction
    final_pred = v6_pred
    
    # Apply targeted adjustments based on patterns observed in testing
    
    # Pattern 1: Medium receipt amounts often benefit from slight boost
    if 200 <= receipts <= 800 and trip_days >= 3:
        # These cases showed systematic underprediction
        boost = min(20, receipts * 0.02)
        final_pred += boost
    
    # Pattern 2: Very short trips with high miles
    if trip_days <= 2 and miles > 400:
        # Rule engine sometimes underestimates these
        if final_pred < miles * 0.5:
            final_pred = miles * 0.5 + receipts * 0.3
    
    # Pattern 3: Long trips with low receipts per day
    if trip_days >= 10:
        receipts_per_day = receipts / trip_days
        if receipts_per_day < 100:
            # Apply small correction for underprediction
            final_pred *= 1.03
    
    # Conservative bounds - don't deviate too far from rule engine
    max_deviation = 100  # Maximum $100 deviation from rule engine
    
    if final_pred > v6_pred + max_deviation:
        final_pred = v6_pred + max_deviation
    elif final_pred < v6_pred - max_deviation:
        final_pred = v6_pred - max_deviation
    
    # Apply cents adjustment for special patterns
    # Medium trips (3-5 days) often end in .49
    if 3 <= trip_days <= 5 and receipts < 500:
        if abs(final_pred - int(final_pred) - 0.49) > 0.20:
            final_pred = int(final_pred) + 0.49
    
    # Very short trips often have .29 or .79 endings
    elif trip_days <= 2:
        decimal = final_pred - int(final_pred)
        if 0.15 <= decimal <= 0.35:
            final_pred = int(final_pred) + 0.29
        elif 0.65 <= decimal <= 0.85:
            final_pred = int(final_pred) + 0.79
    
    # High value trips often end in .99
    elif final_pred > 1500:
        if final_pred - int(final_pred) > 0.70:
            final_pred = int(final_pred) + 0.99
    
    return round(final_pred, 2)


def test_v6_ensemble():
    """Test the v6 ensemble model performance"""
    # Load data
    df = pd.read_csv('../public_cases_expected_outputs.csv')
    
    print("V6 ENSEMBLE MODEL PERFORMANCE TEST")
    print("=" * 60)
    
    # Calculate predictions
    predictions = []
    for _, row in df.iterrows():
        pred = calculate_reimbursement_practical_ensemble_v6(
            row['trip_days'], 
            row['miles'], 
            row['receipts']
        )
        predictions.append(pred)
    
    df['v6_pred'] = predictions
    df['error'] = np.abs(df['v6_pred'] - df['expected_output'])
    
    # Calculate metrics
    mae = df['error'].mean()
    bias = (df['v6_pred'] - df['expected_output']).mean()
    rmse = np.sqrt(((df['v6_pred'] - df['expected_output']) ** 2).mean())
    
    print(f"\nOverall Performance:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Bias: ${bias:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    
    # Compare with v5
    print(f"\nComparison with v5:")
    print(f"  v5 MAE: $77.41")
    print(f"  v6 MAE: ${mae:.2f}")
    print(f"  Improvement: ${77.41 - mae:.2f}")
    print(f"  Improvement %: {(77.41 - mae) / 77.41 * 100:.1f}%")
    
    # Check exact matches
    df['exact_match'] = np.abs(df['v6_pred'] - df['expected_output']) < 0.01
    print(f"\nExact matches: {df['exact_match'].sum()}/1000")
    
    # Error distribution
    print(f"\nError Distribution:")
    print(f"  < $50: {(df['error'] < 50).sum()} cases")
    print(f"  < $100: {(df['error'] < 100).sum()} cases")
    print(f"  < $200: {(df['error'] < 200).sum()} cases")
    print(f"  >= $200: {(df['error'] >= 200).sum()} cases")
    
    # Check receipt ending patterns
    print(f"\nReceipt Ending Analysis:")
    for ending in [49, 99]:
        mask = (df['receipts'] * 100 % 100).astype(int) == ending
        if mask.sum() > 0:
            print(f"  Receipts ending in .{ending}: MAE ${df[mask]['error'].mean():.2f} ({mask.sum()} cases)")
    
    # Check for cents patterns in outputs
    df['output_cents'] = ((df['expected_output'] * 100) % 100).astype(int)
    df['pred_cents'] = ((df['v6_pred'] * 100) % 100).astype(int)
    df['cents_match'] = df['output_cents'] == df['pred_cents']
    
    print(f"\nCents pattern matching: {df['cents_match'].sum()}/1000")
    
    # Top cents patterns in outputs
    print(f"\nTop output cents patterns:")
    top_cents = df['output_cents'].value_counts().head(10)
    for cents, count in top_cents.items():
        match_rate = df[df['output_cents'] == cents]['cents_match'].mean() * 100
        print(f"  .{cents:02d}: {count} cases (match rate: {match_rate:.1f}%)")
    
    return df


if __name__ == "__main__":
    df = test_v6_ensemble()
    
    # Save predictions
    print("\nSaving v6 predictions...")
    df[['case_id', 'v6_pred']].to_csv('../predictions/v6_ensemble_predictions.csv', index=False)
    print("Predictions saved to predictions/v6_ensemble_predictions.csv") 