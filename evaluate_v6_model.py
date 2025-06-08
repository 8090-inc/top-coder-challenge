"""Evaluate V6 XGBoost model and compare with V5"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from model_framework.core.evaluator import ModelEvaluator
from model_framework.models.baseline_models import V5Model
from model_framework.models.v6_xgboost_breakthrough import V6_XGBoostBreakthroughModel
from models.v5_practical_ensemble import calculate_reimbursement_v5


def cross_validate_v6():
    """Perform cross-validation on V6 model"""
    
    print("=== V6 MODEL CROSS-VALIDATION ===\n")
    
    # Load data
    df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # K-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    mae_scores = []
    mape_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"Fold {fold + 1}/5...")
        
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        # Train model
        model = V6_XGBoostBreakthroughModel()
        model.train(train_data)
        
        # Evaluate on test set
        errors = []
        percent_errors = []
        
        for _, row in test_data.iterrows():
            pred = model.predict(row['trip_days'], row['miles'], row['receipts'])
            actual = row['expected_output']
            
            error = abs(pred - actual)
            percent_error = error / actual * 100
            
            errors.append(error)
            percent_errors.append(percent_error)
        
        mae = np.mean(errors)
        mape = np.mean(percent_errors)
        
        mae_scores.append(mae)
        mape_scores.append(mape)
        
        print(f"  Fold {fold + 1} MAE: ${mae:.2f}, MAPE: {mape:.1f}%")
    
    print(f"\nCross-validation results:")
    print(f"  Average MAE: ${np.mean(mae_scores):.2f} (+/- ${np.std(mae_scores):.2f})")
    print(f"  Average MAPE: {np.mean(mape_scores):.1f}% (+/- {np.std(mape_scores):.1f}%)")
    
    return np.mean(mae_scores)


def compare_v5_v6():
    """Compare V5 and V6 models"""
    
    print("\n=== MODEL COMPARISON: V5 vs V6 ===\n")
    
    # Load data
    df = pd.read_csv('public_cases_expected_outputs.csv')
    
    # Train V6 on full dataset
    print("Training V6 on full dataset...")
    v6_model = V6_XGBoostBreakthroughModel()
    v6_model.train(df)
    
    # Evaluate both models
    v5_errors = []
    v6_errors = []
    improvements = []
    
    print("\nEvaluating on training data...")
    
    for _, row in df.iterrows():
        actual = row['expected_output']
        
        # V5 prediction
        v5_pred = calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts'])
        v5_error = abs(v5_pred - actual)
        v5_errors.append(v5_error)
        
        # V6 prediction
        v6_pred = v6_model.predict(row['trip_days'], row['miles'], row['receipts'])
        v6_error = abs(v6_pred - actual)
        v6_errors.append(v6_error)
        
        # Track improvement
        improvement = v5_error - v6_error
        improvements.append(improvement)
    
    # Calculate metrics
    v5_mae = np.mean(v5_errors)
    v6_mae = np.mean(v6_errors)
    
    v5_max_error = np.max(v5_errors)
    v6_max_error = np.max(v6_errors)
    
    v5_errors_over_500 = sum(1 for e in v5_errors if e > 500)
    v6_errors_over_500 = sum(1 for e in v6_errors if e > 500)
    
    improved_cases = sum(1 for i in improvements if i > 0)
    worsened_cases = sum(1 for i in improvements if i < 0)
    
    print(f"\nResults:")
    print(f"  V5 MAE: ${v5_mae:.2f}")
    print(f"  V6 MAE: ${v6_mae:.2f}")
    print(f"  Improvement: ${v5_mae - v6_mae:.2f} ({(v5_mae - v6_mae) / v5_mae * 100:.1f}%)")
    print(f"\n  V5 Max Error: ${v5_max_error:.2f}")
    print(f"  V6 Max Error: ${v6_max_error:.2f}")
    print(f"\n  V5 Errors > $500: {v5_errors_over_500}")
    print(f"  V6 Errors > $500: {v6_errors_over_500}")
    print(f"\n  Cases improved: {improved_cases}/{len(df)} ({improved_cases/len(df)*100:.1f}%)")
    print(f"  Cases worsened: {worsened_cases}/{len(df)} ({worsened_cases/len(df)*100:.1f}%)")
    
    # Show biggest improvements
    improvement_df = pd.DataFrame({
        'trip_days': df['trip_days'],
        'miles': df['miles'],
        'receipts': df['receipts'],
        'actual': df['expected_output'],
        'v5_pred': [calculate_reimbursement_v5(row['trip_days'], row['miles'], row['receipts']) for _, row in df.iterrows()],
        'v6_pred': [v6_model.predict(row['trip_days'], row['miles'], row['receipts']) for _, row in df.iterrows()],
        'v5_error': v5_errors,
        'v6_error': v6_errors,
        'improvement': improvements
    })
    
    print("\n=== BIGGEST IMPROVEMENTS ===")
    top_improvements = improvement_df.nlargest(10, 'improvement')
    for _, row in top_improvements.iterrows():
        print(f"  Days={row['trip_days']}, Miles={row['miles']:.0f}, Receipts=${row['receipts']:.2f}")
        print(f"    Actual: ${row['actual']:.2f}, V5: ${row['v5_pred']:.2f} (err ${row['v5_error']:.2f}), V6: ${row['v6_pred']:.2f} (err ${row['v6_error']:.2f})")
        print(f"    Improvement: ${row['improvement']:.2f}")
    
    # Show biggest worsenings
    print("\n=== BIGGEST WORSENINGS ===")
    worst_cases = improvement_df.nsmallest(10, 'improvement')
    for _, row in worst_cases.iterrows():
        print(f"  Days={row['trip_days']}, Miles={row['miles']:.0f}, Receipts=${row['receipts']:.2f}")
        print(f"    Actual: ${row['actual']:.2f}, V5: ${row['v5_pred']:.2f} (err ${row['v5_error']:.2f}), V6: ${row['v6_pred']:.2f} (err ${row['v6_error']:.2f})")
        print(f"    Worsened by: ${-row['improvement']:.2f}")
    
    return v6_mae


def analyze_feature_importance(model):
    """Analyze which features are most important"""
    
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    if model.models and 'xgb' in model.models:
        importance = pd.DataFrame({
            'feature': model.feature_names,
            'importance': model.models['xgb'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        for i, (_, row) in enumerate(importance.head(20).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Group by feature type
        print("\nImportance by feature type:")
        
        core_importance = importance[importance['feature'].isin(['trip_days', 'miles', 'receipts'])]['importance'].sum()
        print(f"  Core features: {core_importance:.3f}")
        
        prediction_features = ['v3_prediction', 'v5_prediction', 'prediction_diff', 'prediction_ratio']
        pred_importance = importance[importance['feature'].isin(prediction_features)]['importance'].sum()
        print(f"  Prediction features: {pred_importance:.3f}")
        
        interaction_features = ['days_miles', 'days_receipts', 'miles_receipts', 'triple_interaction']
        int_importance = importance[importance['feature'].isin(interaction_features)]['importance'].sum()
        print(f"  Interaction features: {int_importance:.3f}")


if __name__ == "__main__":
    # Run cross-validation
    cv_mae = cross_validate_v6()
    
    # Compare models
    v6_mae = compare_v5_v6()
    
    # Train final model for feature analysis
    df = pd.read_csv('public_cases_expected_outputs.csv')
    final_model = V6_XGBoostBreakthroughModel()
    final_model.train(df)
    
    # Analyze features
    analyze_feature_importance(final_model)
    
    print(f"\n=== SUMMARY ===")
    print(f"V5 Baseline MAE: $77.41 (from hypothesis doc)")
    print(f"V6 Training MAE: ${v6_mae:.2f}")
    print(f"V6 Cross-Val MAE: ${cv_mae:.2f}")
    
    if cv_mae < 77.41:
        improvement = (77.41 - cv_mae) / 77.41 * 100
        print(f"\n🎉 BREAKTHROUGH! V6 improves by {improvement:.1f}% over V5!")
    else:
        print(f"\n⚠️  V6 did not improve over V5 baseline") 