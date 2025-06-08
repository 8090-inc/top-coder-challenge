"""Advanced cents pattern discovery using ML and pattern mining"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def create_cents_features(df):
    """Create comprehensive features for cents prediction"""
    
    features = pd.DataFrame()
    
    # Basic features
    features['trip_days'] = df['trip_days']
    features['miles'] = df['miles']
    features['receipts'] = df['receipts']
    features['receipt_cents'] = (df['receipts'] * 100).astype(int) % 100
    
    # Ratios
    features['miles_per_day'] = df['miles'] / df['trip_days']
    features['receipts_per_day'] = df['receipts'] / df['trip_days']
    features['receipts_per_mile'] = df['receipts'] / (df['miles'] + 1)
    
    # Integer features
    features['miles_int'] = df['miles'].astype(int)
    features['receipts_int'] = df['receipts'].astype(int)
    features['receipts_total_cents'] = (df['receipts'] * 100).astype(int)
    
    # Modulo features
    for mod in [7, 10, 12, 24, 30, 60, 100]:
        features[f'days_mod_{mod}'] = df['trip_days'] % mod
        features[f'miles_mod_{mod}'] = df['miles'].astype(int) % mod
        features[f'receipts_cents_mod_{mod}'] = features['receipts_total_cents'] % mod
    
    # Combined features
    features['sum_all'] = df['trip_days'] + df['miles'] + df['receipts']
    features['sum_int'] = df['trip_days'] + df['miles'].astype(int) + df['receipts'].astype(int)
    features['product_days_miles'] = df['trip_days'] * df['miles']
    features['product_days_receipts'] = df['trip_days'] * df['receipts']
    
    # Digit features
    features['days_digits'] = df['trip_days'].apply(lambda x: sum(int(d) for d in str(int(x))))
    features['miles_digits'] = df['miles'].apply(lambda x: sum(int(d) for d in str(int(x))))
    features['receipts_digits'] = df['receipts'].apply(lambda x: sum(int(d) for d in str(int(x * 100))))
    features['all_digits'] = features['days_digits'] + features['miles_digits'] + features['receipts_digits']
    
    # Binary features
    features['is_single_day'] = (df['trip_days'] == 1).astype(int)
    features['is_long_trip'] = (df['trip_days'] >= 10).astype(int)
    features['is_high_miles'] = (df['miles'] > 1000).astype(int)
    features['is_high_receipts'] = (df['receipts'] > 1000).astype(int)
    features['receipt_ends_49'] = (features['receipt_cents'] == 49).astype(int)
    features['receipt_ends_99'] = (features['receipt_cents'] == 99).astype(int)
    
    # Cluster features (from hypothesis)
    features['cluster_0'] = ((df['trip_days'] > 1) & (df['miles'] < 2000) & (df['receipts'] < 1800)).astype(int)
    features['cluster_1'] = ((df['trip_days'] == 1) & (df['miles'] >= 600)).astype(int)
    features['cluster_2'] = (df['trip_days'] >= 10).astype(int)
    features['cluster_5'] = ((df['trip_days'] >= 7) & (df['trip_days'] <= 8) & 
                             (df['miles'] >= 900) & (df['miles'] <= 1200)).astype(int)
    
    return features

def train_cents_models(X, y):
    """Train various models to predict cents"""
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, max_depth=20, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, random_state=42),
        'NeuralNet': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"  CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Fit on full data
        model.fit(X, y)
        
        # Feature importance for tree models
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"  Top 10 features:")
            for _, row in importances.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
        
        results[name] = model
    
    return results

def analyze_rule_patterns(df):
    """Look for deterministic rules in the data"""
    
    print("\n=== SEARCHING FOR DETERMINISTIC RULES ===")
    
    # Add output cents
    df['output_cents'] = (df['expected_output'] * 100).astype(int) % 100
    
    # Look for exact relationships
    df['dollar_amount'] = df['expected_output'].astype(int)
    
    # Check if cents are related to dollar amount
    print("\nChecking if cents relate to dollar amount...")
    for cents in df['output_cents'].unique()[:10]:
        subset = df[df['output_cents'] == cents]
        if len(subset) > 5:
            dollar_pattern = subset['dollar_amount'] % 100
            if len(dollar_pattern.unique()) <= 3:
                print(f"  Cents {cents}: dollar_amount % 100 = {dollar_pattern.unique()}")
    
    # Check mathematical relationships
    print("\nChecking mathematical relationships...")
    
    # Test if output = f(base_calculation)
    # Hypothesis: cents might be derived from the base calculation before rounding
    
    # Approximate base calculation (from v3 model logic)
    df['approx_base'] = 100 + 50 * df['trip_days'] + 0.4 * df['miles'] + 0.5 * df['receipts']
    df['approx_base_cents'] = (df['approx_base'] * 100).astype(int) % 100
    
    # Check correlation
    matches = (df['approx_base_cents'] == df['output_cents']).sum()
    print(f"  Approximate base cents match: {matches}/{len(df)} ({matches/len(df)*100:.1f}%)")
    
    # Look for patterns in specific clusters
    print("\nCluster-specific patterns:")
    
    # Single day trips
    single_day = df[df['trip_days'] == 1]
    print(f"  Single day trips: {len(single_day)} cases")
    print(f"    Common cents: {single_day['output_cents'].value_counts().head(5).to_dict()}")
    
    # Long trips
    long_trips = df[df['trip_days'] >= 10]
    print(f"  Long trips (10+ days): {len(long_trips)} cases")
    print(f"    Common cents: {long_trips['output_cents'].value_counts().head(5).to_dict()}")

def main():
    """Main analysis function"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Create features
    print("Creating features...")
    X = create_cents_features(df)
    y = (df['expected_output'] * 100).astype(int) % 100
    
    # Train models
    models = train_cents_models(X, y)
    
    # Analyze patterns
    analyze_rule_patterns(df)
    
    # Make predictions with best model
    print("\n=== TESTING PREDICTIONS ===")
    best_model = models['ExtraTrees']  # Usually performs best
    
    # Predict on training data to see accuracy
    y_pred = best_model.predict(X)
    exact_matches = (y_pred == y).sum()
    print(f"\nExact cents matches: {exact_matches}/{len(y)} ({exact_matches/len(y)*100:.1f}%)")
    
    # Show some examples
    print("\nExample predictions:")
    for i in range(10):
        if y_pred[i] == y.iloc[i]:
            print(f"  MATCH: Days={df.iloc[i]['trip_days']}, Miles={df.iloc[i]['miles']:.0f}, "
                  f"Receipts=${df.iloc[i]['receipts']:.2f} -> Cents={y.iloc[i]}")
    
    # Save the best model
    import joblib
    joblib.dump(best_model, 'cents_predictor_model.pkl')
    print("\nBest model saved as 'cents_predictor_model.pkl'")
    
    return best_model

if __name__ == "__main__":
    model = main() 