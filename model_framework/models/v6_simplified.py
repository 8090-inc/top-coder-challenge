"""V6 Simplified - XGBoost + LightGBM ensemble for breakthrough performance"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from model_framework.core.base_model import BaseModel
from models.v5_practical_ensemble import calculate_reimbursement_v5
from models.cluster_models_optimized import calculate_reimbursement_v3


class V6_SimplifiedModel(BaseModel):
    """V6 Simplified - XGBoost + LightGBM with advanced features"""
    
    def __init__(self):
        super().__init__(
            model_id="xgb_lgb_ensemble",
            version="6.0",
            description="Simplified XGBoost + LightGBM ensemble with advanced features"
        )
        self.models = {}
        self.feature_names = None
        self.is_trained = False
        
    def create_features(self, trip_days, miles, receipts):
        """Create advanced features"""
        
        features = {}
        
        # Core features
        features['trip_days'] = trip_days
        features['miles'] = miles
        features['receipts'] = receipts
        
        # Get v3 and v5 predictions
        v3_pred = calculate_reimbursement_v3(trip_days, miles, receipts)
        v5_pred = calculate_reimbursement_v5(trip_days, miles, receipts)
        
        features['v3_prediction'] = v3_pred
        features['v5_prediction'] = v5_pred
        features['prediction_diff'] = v5_pred - v3_pred
        features['prediction_avg'] = (v3_pred + v5_pred) / 2
        
        # Efficiency metrics
        features['miles_per_day'] = miles / max(trip_days, 1)
        features['receipts_per_day'] = receipts / max(trip_days, 1)
        features['receipts_per_mile'] = receipts / max(miles, 1)
        
        # Log features
        features['log_miles'] = np.log1p(miles)
        features['log_receipts'] = np.log1p(receipts)
        features['log_days'] = np.log1p(trip_days)
        
        # Interaction features
        features['days_miles'] = trip_days * miles
        features['days_receipts'] = trip_days * receipts
        features['miles_receipts'] = miles * receipts
        
        # Power features
        features['miles_squared'] = miles ** 2
        features['receipts_squared'] = receipts ** 2
        features['sqrt_miles'] = np.sqrt(miles)
        features['sqrt_receipts'] = np.sqrt(receipts)
        
        # Receipt pattern features
        receipt_cents = int(receipts * 100) % 100
        features['receipt_cents'] = receipt_cents
        features['receipt_ends_49'] = int(receipt_cents == 49)
        features['receipt_ends_99'] = int(receipt_cents == 99)
        
        # Cluster indicators
        features['is_single_day'] = int(trip_days == 1)
        features['is_long_trip'] = int(trip_days >= 10)
        features['is_high_miles'] = int(miles > 1000)
        features['is_high_receipts'] = int(receipts > 1000)
        features['is_kevin_special'] = int(7 <= trip_days <= 8 and 900 <= miles <= 1200)
        
        # Binned features
        features['efficiency_bin'] = np.digitize(miles/max(trip_days,1), [0, 100, 200, 300, 400])
        features['receipt_bin'] = np.digitize(receipts, [0, 100, 500, 1000, 2000])
        
        return features
    
    def train(self, train_data):
        """Train the ensemble"""
        print("Training V6 Simplified Model...")
        
        # Prepare features
        X_list = []
        y_list = []
        
        for _, row in train_data.iterrows():
            features = self.create_features(row['trip_days'], row['miles'], row['receipts'])
            X_list.append(features)
            y_list.append(row['expected_output'])
        
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        self.feature_names = X.columns.tolist()
        
        # Train XGBoost
        print("Training XGBoost...")
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            objective='reg:absoluteerror'
        )
        self.models['xgb'].fit(X, y)
        
        # Train LightGBM
        print("Training LightGBM...")
        self.models['lgb'] = lgb.LGBMRegressor(
            n_estimators=1500,
            num_leaves=50,
            learning_rate=0.03,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            objective='mae'
        )
        self.models['lgb'].fit(X, y)
        
        # Train blending weights
        print("Training blending weights...")
        xgb_pred = self.models['xgb'].predict(X)
        lgb_pred = self.models['lgb'].predict(X)
        
        # Find optimal blend
        best_weight = 0.5
        best_mae = float('inf')
        
        for w in np.arange(0, 1.01, 0.05):
            blend = w * xgb_pred + (1 - w) * lgb_pred
            mae = np.mean(np.abs(blend - y))
            if mae < best_mae:
                best_mae = mae
                best_weight = w
        
        self.xgb_weight = best_weight
        print(f"Optimal blend: {self.xgb_weight:.2f} XGBoost + {1-self.xgb_weight:.2f} LightGBM")
        print(f"Training MAE: ${best_mae:.2f}")
        
        self.is_trained = True
        
        # Feature importance
        print("\nTop 15 features (XGBoost):")
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models['xgb'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, trip_days, miles, receipts):
        """Make prediction"""
        
        if not self.is_trained:
            return calculate_reimbursement_v5(trip_days, miles, receipts)
        
        # Create features
        features = self.create_features(trip_days, miles, receipts)
        X = pd.DataFrame([features])[self.feature_names]
        
        # Get predictions
        xgb_pred = self.models['xgb'].predict(X)[0]
        lgb_pred = self.models['lgb'].predict(X)[0]
        
        # Blend
        prediction = self.xgb_weight * xgb_pred + (1 - self.xgb_weight) * lgb_pred
        
        # Apply receipt penalties
        receipt_cents = int(receipts * 100) % 100
        if receipt_cents == 49:
            prediction *= 0.341
        elif receipt_cents == 99:
            prediction *= 0.51
        
        return round(max(0, prediction), 2) 