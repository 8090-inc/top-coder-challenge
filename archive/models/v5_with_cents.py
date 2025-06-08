import pandas as pd
import numpy as np
import joblib
from models.practical_ensemble import PracticalEnsemble
from models.cents_classifier import train_cents_classifier, predict_cents

class V5WithCents:
    """Enhanced v5 model with cents prediction"""
    
    def __init__(self):
        self.ensemble = None
        self.cents_clf = None
        
    def train(self, train_df):
        """Train both the ensemble and cents classifier"""
        # Train the v5 ensemble
        self.ensemble = PracticalEnsemble()
        self.ensemble.train(train_df)
        
        # Add cluster assignments for cents classifier
        train_df_with_clusters = train_df.copy()
        for idx, row in train_df_with_clusters.iterrows():
            cluster = self.ensemble.rule_engine.assign_cluster(
                row['trip_days'], row['miles'], row['receipts']
            )
            train_df_with_clusters.loc[idx, 'cluster'] = cluster
        
        # Train cents classifier
        self.cents_clf = train_cents_classifier(train_df_with_clusters)
        
    def predict(self, trip_days, miles, receipts):
        """Predict with dollar amount from ensemble and cents from classifier"""
        # Get base prediction from v5 ensemble
        base_prediction = self.ensemble.predict(trip_days, miles, receipts)
        
        # Get cluster for cents prediction
        cluster = self.ensemble.rule_engine.assign_cluster(trip_days, miles, receipts)
        
        # Predict cents
        predicted_cents = predict_cents(self.cents_clf, trip_days, miles, receipts, cluster)
        
        # Combine dollar and cents predictions
        dollars = int(base_prediction)
        final_prediction = dollars + (predicted_cents / 100.0)
        
        return final_prediction
    
    def evaluate(self, test_df):
        """Evaluate model performance"""
        predictions = []
        
        for _, row in test_df.iterrows():
            pred = self.predict(row['trip_days'], row['miles'], row['receipts'])
            predictions.append(pred)
        
        test_df['predicted'] = predictions
        test_df['error'] = np.abs(test_df['predicted'] - test_df['expected_output'])
        
        mae = test_df['error'].mean()
        mape = (test_df['error'] / test_df['expected_output'] * 100).mean()
        max_error = test_df['error'].max()
        
        print(f"V5 with Cents Performance:")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.1f}%")
        print(f"Max Error: ${max_error:.2f}")
        
        # Check exact matches
        exact_matches = (test_df['predicted'] == test_df['expected_output']).sum()
        print(f"Exact matches: {exact_matches}/{len(test_df)} ({exact_matches/len(test_df)*100:.1f}%)")
        
        return mae, test_df 