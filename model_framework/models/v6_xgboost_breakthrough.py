"""V6 XGBoost Breakthrough Model - Advanced techniques for massive improvement"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
from catboost import CatBoostRegressor
from model_framework.core.base_model import BaseModel
from models.v5_practical_ensemble import calculate_reimbursement_v5
from models.cluster_models_optimized import calculate_reimbursement_v3


class V6_XGBoostBreakthroughModel(BaseModel):
    """V6 - Advanced XGBoost with sophisticated feature engineering and ensemble"""
    
    def __init__(self):
        super().__init__(
            model_id="xgboost_breakthrough",
            version="6.0",
            description="Advanced XGBoost with feature engineering and multi-stage prediction"
        )
        self.models = {}
        self.feature_names = None
        self.is_trained = False
        
    def create_advanced_features(self, trip_days, miles, receipts):
        """Create sophisticated features for better prediction"""
        
        features = {}
        
        # === Core Features ===
        features['trip_days'] = trip_days
        features['miles'] = miles
        features['receipts'] = receipts
        
        # === Efficiency Metrics ===
        features['miles_per_day'] = miles / max(trip_days, 1)
        features['receipts_per_day'] = receipts / max(trip_days, 1)
        features['receipts_per_mile'] = receipts / max(miles, 1)
        features['efficiency_ratio'] = miles / (receipts + 1)
        
        # === Logarithmic Features (capture non-linear relationships) ===
        features['log_miles'] = np.log1p(miles)
        features['log_receipts'] = np.log1p(receipts)
        features['log_days'] = np.log1p(trip_days)
        
        # === Power Features ===
        features['miles_squared'] = miles ** 2
        features['receipts_squared'] = receipts ** 2
        features['days_squared'] = trip_days ** 2
        features['sqrt_miles'] = np.sqrt(miles)
        features['sqrt_receipts'] = np.sqrt(receipts)
        
        # === Interaction Features ===
        features['days_miles'] = trip_days * miles
        features['days_receipts'] = trip_days * receipts
        features['miles_receipts'] = miles * receipts
        features['triple_interaction'] = trip_days * miles * receipts
        
        # === Ratio Features ===
        features['miles_to_receipts_ratio'] = miles / (receipts + 1)
        features['days_to_miles_ratio'] = trip_days / (miles + 1)
        features['days_to_receipts_ratio'] = trip_days / (receipts + 1)
        
        # === Binned Features ===
        features['miles_bin'] = np.digitize(miles, [0, 100, 300, 600, 1000, 2000])
        features['receipts_bin'] = np.digitize(receipts, [0, 50, 200, 500, 1000, 2000])
        features['efficiency_bin'] = np.digitize(miles/max(trip_days,1), [0, 50, 150, 250, 400])
        
        # === Cluster Features (from hypothesis) ===
        features['is_standard'] = int(trip_days > 1 and miles < 2000 and receipts < 1800)
        features['is_single_day_high_miles'] = int(trip_days == 1 and miles >= 600)
        features['is_long_trip'] = int(trip_days >= 10)
        features['is_short_intense'] = int(3 <= trip_days <= 5 and receipts > 1500)
        features['is_kevin_special'] = int(7 <= trip_days <= 8 and 900 <= miles <= 1200 and 1000 <= receipts <= 1200)
        
        # === Receipt Pattern Features ===
        receipt_cents = int(receipts * 100) % 100
        features['receipt_cents'] = receipt_cents
        features['receipt_ends_49'] = int(receipt_cents == 49)
        features['receipt_ends_99'] = int(receipt_cents == 99)
        features['receipt_round'] = int(receipt_cents == 0)
        
        # === Modulo Features (might capture hidden patterns) ===
        features['days_mod_7'] = trip_days % 7
        features['miles_mod_100'] = int(miles) % 100
        features['receipts_mod_50'] = int(receipts) % 50
        
        # === Statistical Features ===
        features['cv_features'] = np.std([trip_days, miles/100, receipts/100]) / (np.mean([trip_days, miles/100, receipts/100]) + 1)
        features['skewness'] = (3 * (np.mean([trip_days, miles/100, receipts/100]) - np.median([trip_days, miles/100, receipts/100]))) / (np.std([trip_days, miles/100, receipts/100]) + 1)
        
        # === Rule Engine Predictions as Features ===
        features['v3_prediction'] = calculate_reimbursement_v3(trip_days, miles, receipts)
        features['v5_prediction'] = calculate_reimbursement_v5(trip_days, miles, receipts)
        features['prediction_diff'] = features['v5_prediction'] - features['v3_prediction']
        features['prediction_ratio'] = features['v5_prediction'] / (features['v3_prediction'] + 1)
        
        # === Polynomial Combinations ===
        features['poly_1'] = trip_days ** 2 + miles + receipts
        features['poly_2'] = trip_days + miles ** 2 + receipts
        features['poly_3'] = trip_days + miles + receipts ** 2
        
        return features
    
    def train(self, train_data):
        """Train the ensemble model"""
        print("Training V6 XGBoost Breakthrough Model...")
        
        # Prepare features
        X_list = []
        y_list = []
        
        for _, row in train_data.iterrows():
            features = self.create_advanced_features(row['trip_days'], row['miles'], row['receipts'])
            X_list.append(features)
            y_list.append(row['expected_output'])
        
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        self.feature_names = X.columns.tolist()
        
        # === Multi-Model Ensemble ===
        
        # 1. XGBoost with custom objective
        print("Training XGBoost...")
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            objective='reg:absoluteerror',  # MAE objective
            tree_method='hist',
            enable_categorical=True
        )
        
        # 2. LightGBM
        print("Training LightGBM...")
        self.models['lgb'] = lgb.LGBMRegressor(
            n_estimators=2000,
            num_leaves=64,
            learning_rate=0.02,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            objective='mae'
        )
        
        # 3. CatBoost
        print("Training CatBoost...")
        self.models['cat'] = CatBoostRegressor(
            iterations=1500,
            depth=8,
            learning_rate=0.02,
            l2_leaf_reg=3,
            loss_function='MAE',
            random_seed=42,
            verbose=False
        )
        
        # Train all models
        for name, model in self.models.items():
            model.fit(X, y)
        
        # === Second Stage: Meta-learner ===
        print("Training meta-learner...")
        
        # Get predictions from base models for stacking
        base_preds = np.column_stack([
            model.predict(X) for model in self.models.values()
        ])
        
        # Add base predictions as features
        X_meta = np.column_stack([X, base_preds])
        
        # Train meta-learner
        self.meta_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            objective='reg:absoluteerror'
        )
        self.meta_model.fit(X_meta, y)
        
        self.is_trained = True
        
        # Feature importance analysis
        print("\nTop 20 most important features (XGBoost):")
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models['xgb'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, trip_days, miles, receipts):
        """Make prediction using the ensemble"""
        
        if not self.is_trained:
            # Fallback to v5 if not trained
            return calculate_reimbursement_v5(trip_days, miles, receipts)
        
        # Create features
        features = self.create_advanced_features(trip_days, miles, receipts)
        X = pd.DataFrame([features])[self.feature_names]
        
        # Get base model predictions
        base_preds = np.array([
            model.predict(X)[0] for model in self.models.values()
        ])
        
        # Prepare meta features
        X_meta = np.column_stack([X, base_preds.reshape(1, -1)])
        
        # Get final prediction
        prediction = self.meta_model.predict(X_meta)[0]
        
        # Apply business rules
        # Ensure non-negative
        prediction = max(0, prediction)
        
        # Apply receipt ending penalties (from hypothesis)
        receipt_cents = int(receipts * 100) % 100
        if receipt_cents == 49:
            prediction *= 0.341
        elif receipt_cents == 99:
            prediction *= 0.51
        
        return round(prediction, 2)


def create_submission_model():
    """Create and return the model for submission"""
    return V6_XGBoostBreakthroughModel()


if __name__ == "__main__":
    # Test the model
    model = create_submission_model()
    
    # Load training data
    import pandas as pd
    train_data = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Train the model
    model.train(train_data)
    
    # Test predictions
    print("\nSample predictions:")
    for i in range(5):
        row = train_data.iloc[i]
        pred = model.predict(row['trip_days'], row['miles'], row['receipts'])
        actual = row['expected_output']
        error = abs(pred - actual)
        print(f"Case {i}: Predicted ${pred:.2f}, Actual ${actual:.2f}, Error ${error:.2f}") 