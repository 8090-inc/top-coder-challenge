import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import joblib

def train_cents_classifier(df):
    """Train a classifier to predict exact cents values"""
    
    # Extract cents from expected output
    df['cents_target'] = (df['expected_output'] * 100).round().astype(int) % 100
    
    # Create features
    features = pd.DataFrame()
    features['trip_days'] = df['trip_days']
    features['miles'] = df['miles']
    features['receipts_dollars'] = df['receipts'].astype(int)
    features['receipts_cents'] = (df['receipts'] * 100).astype(int) % 100
    features['miles_per_day'] = df['miles'] / df['trip_days'].clip(lower=1)
    features['receipts_per_day'] = df['receipts'] / df['trip_days'].clip(lower=1)
    
    # Add modular features
    features['days_mod_7'] = df['trip_days'] % 7
    features['miles_mod_100'] = df['miles'].astype(int) % 100
    features['receipts_mod_100'] = df['receipts'].astype(int) % 100
    
    # Add cluster assignment
    features['cluster'] = df['cluster'] if 'cluster' in df else 0
    
    # Train classifier
    clf = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    X = features
    y = df['cents_target']
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"Cents classifier CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    # Fit on all data
    clf.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop features for cents prediction:")
    print(importance.head(10))
    
    return clf

def predict_cents(clf, trip_days, miles, receipts, cluster=0):
    """Predict cents value for a single case"""
    features = pd.DataFrame({
        'trip_days': [trip_days],
        'miles': [miles],
        'receipts_dollars': [int(receipts)],
        'receipts_cents': [int(receipts * 100) % 100],
        'miles_per_day': [miles / max(trip_days, 1)],
        'receipts_per_day': [receipts / max(trip_days, 1)],
        'days_mod_7': [int(trip_days) % 7],
        'miles_mod_100': [int(miles) % 100],
        'receipts_mod_100': [int(receipts) % 100],
        'cluster': [cluster]
    })
    
    return clf.predict(features)[0] 