#!/usr/bin/env python3
"""
Comprehensive Hypothesis Testing for Legacy Reimbursement System
Phase 2: Formula Component Discovery

This script systematically tests all hypotheses from employee interviews
and discovers the complete mathematical formula.
"""

import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFormulaDiscovery:
    """Complete formula discovery for the legacy reimbursement system"""
    
    def __init__(self):
        self.df = None
        self.formula_components = {}
        self.load_data()
        
    def load_data(self):
        """Load the historical data"""
        print("📊 Loading 1,000 historical cases...")
        
        with open('public_cases.json', 'r') as f:
            raw_data = json.load(f)
            
        data_rows = []
        for case in raw_data:
            row = {
                'days': case['input']['trip_duration_days'],
                'miles': case['input']['miles_traveled'], 
                'receipts': case['input']['total_receipts_amount'],
                'reimbursement': case['expected_output']
            }
            data_rows.append(row)
            
        self.df = pd.DataFrame(data_rows)
        print(f"✅ Loaded {len(self.df)} cases")
        
    def discover_base_formula_structure(self):
        """Discover the fundamental structure of the formula"""
        print("\n🔬 DISCOVERING FORMULA STRUCTURE")
        print("=" * 60)
        
        # Test multiple formula structures
        print("\n📐 Testing Formula Structures:")
        
        # Structure 1: Linear combination
        X_linear = self.df[['days', 'miles', 'receipts']]
        y = self.df['reimbursement']
        
        lr = LinearRegression()
        lr.fit(X_linear, y)
        y_pred_linear = lr.predict(X_linear)
        
        mae_linear = mean_absolute_error(y, y_pred_linear)
        r2_linear = r2_score(y, y_pred_linear)
        
        print(f"  Linear: MAE=${mae_linear:.2f}, R²={r2_linear:.3f}")
        print(f"    Formula: {lr.intercept_:.2f} + {lr.coef_[0]:.2f}*days + {lr.coef_[1]:.3f}*miles + {lr.coef_[2]:.3f}*receipts")
        
        # Store the best linear model
        self.best_linear_model = lr
        self.best_features = ['days', 'miles', 'receipts']
        print(f"  ✅ Best structure: Linear (MAE=${mae_linear:.2f})")
            
        return {'linear_mae': mae_linear}
        
    def discover_component_based_formula(self):
        """Discover formula using component-based approach"""
        print("\n🧩 COMPONENT-BASED FORMULA DISCOVERY")
        print("=" * 60)
        
        # Test the hypothesis: reimbursement = base_per_diem + mileage_component + receipt_component + bonuses
        
        # Step 1: Estimate base per-diem component
        print("\n1️⃣ Base Per-Diem Component:")
        
        # Find minimum reimbursement rate per day (this should be close to base rate)
        min_per_day_by_duration = self.df.groupby('days')['reimbursement'].min() / self.df.groupby('days')['days'].first()
        base_rate = min_per_day_by_duration.min()
        
        print(f"   Estimated base rate: ${base_rate:.2f}/day")
        
        # Calculate base component for each case
        self.df['base_component'] = self.df['days'] * base_rate
        self.df['remaining_after_base'] = self.df['reimbursement'] - self.df['base_component']
        
        print(f"   Average remaining after base: ${self.df['remaining_after_base'].mean():.2f}")
        
        # Step 2: Estimate mileage component
        print("\n2️⃣ Mileage Component:")
        
        # Test tiered mileage hypothesis
        def test_mileage_tiers():
            # Try different threshold values
            best_mae = float('inf')
            best_config = None
            
            for threshold in [50, 100, 150, 200, 300]:
                for rate1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    for rate2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        if rate2 > rate1:
                            continue
                            
                        # Calculate mileage component with this configuration
                        mileage_comp = np.where(self.df['miles'] <= threshold,
                                              self.df['miles'] * rate1,
                                              threshold * rate1 + (self.df['miles'] - threshold) * rate2)
                        
                        remaining_after_mileage = self.df['remaining_after_base'] - mileage_comp
                        mae = mean_absolute_error(remaining_after_mileage, [remaining_after_mileage.mean()] * len(remaining_after_mileage))
                        
                        if mae < best_mae:
                            best_mae = mae
                            best_config = {'threshold': threshold, 'rate1': rate1, 'rate2': rate2, 'mae': mae}
            
            return best_config
        
        mileage_config = test_mileage_tiers()
        print(f"   Best mileage config: {mileage_config}")
        
        # Apply best mileage configuration
        threshold = mileage_config['threshold']
        rate1 = mileage_config['rate1']
        rate2 = mileage_config['rate2']
        
        self.df['mileage_component'] = np.where(self.df['miles'] <= threshold,
                                              self.df['miles'] * rate1,
                                              threshold * rate1 + (self.df['miles'] - threshold) * rate2)
        
        self.df['remaining_after_mileage'] = self.df['remaining_after_base'] - self.df['mileage_component']
        
        print(f"   Average remaining after mileage: ${self.df['remaining_after_mileage'].mean():.2f}")
        
        # Step 3: Estimate receipt component
        print("\n3️⃣ Receipt Component:")
        
        # Test receipt processing hypothesis - look for relationship between receipts and remaining amount
        receipt_correlation = self.df[['receipts', 'remaining_after_mileage']].corr().iloc[0, 1]
        print(f"   Receipt correlation with remaining: {receipt_correlation:.3f}")
        
        # Try linear receipt processing
        if abs(receipt_correlation) > 0.3:
            receipt_rate = self.df['remaining_after_mileage'].sum() / self.df['receipts'].sum()
            print(f"   Estimated receipt rate: {receipt_rate:.3f}x")
            
            self.df['receipt_component'] = self.df['receipts'] * receipt_rate
            self.df['final_residual'] = self.df['remaining_after_mileage'] - self.df['receipt_component']
        else:
            # Try diminishing returns model
            def receipt_model(receipts, scale, power):
                return scale * (receipts ** power)
            
            try:
                popt, _ = curve_fit(receipt_model, self.df['receipts'], self.df['remaining_after_mileage'], 
                                  p0=[0.5, 0.8], bounds=([0, 0], [2, 2]))
                
                scale, power = popt
                print(f"   Receipt model: {scale:.3f} * receipts^{power:.3f}")
                
                self.df['receipt_component'] = scale * (self.df['receipts'] ** power)
                self.df['final_residual'] = self.df['remaining_after_mileage'] - self.df['receipt_component']
                
                receipt_rate = None  # Using power model instead
                
            except:
                print("   Could not fit receipt model, using linear approximation")
                receipt_rate = 0.3  # Default fallback
                self.df['receipt_component'] = self.df['receipts'] * receipt_rate
                self.df['final_residual'] = self.df['remaining_after_mileage'] - self.df['receipt_component']
        
        print(f"   Average final residual: ${self.df['final_residual'].mean():.2f}")
        print(f"   Final residual std: ${self.df['final_residual'].std():.2f}")
        
        # Step 4: Store formula components
        self.formula_components = {
            'base_per_diem_rate': base_rate,
            'mileage_threshold': threshold,
            'mileage_rate1': rate1,
            'mileage_rate2': rate2,
            'receipt_rate': receipt_rate if 'receipt_rate' in locals() else None,
            'receipt_scale': locals().get('scale'),
            'receipt_power': locals().get('power'),
            'average_bonus': self.df['final_residual'].mean()
        }
        
        return self.formula_components
        
    def calculate_prediction_accuracy(self):
        """Calculate how accurate our discovered formula is"""
        print("\n🎯 TESTING FORMULA ACCURACY")
        print("=" * 60)
        
        # Calculate predictions using discovered formula
        predictions = []
        
        for _, row in self.df.iterrows():
            days = row['days']
            miles = row['miles']
            receipts = row['receipts']
            
            # Base component
            base = days * self.formula_components['base_per_diem_rate']
            
            # Mileage component
            threshold = self.formula_components['mileage_threshold']
            rate1 = self.formula_components['mileage_rate1']
            rate2 = self.formula_components['mileage_rate2']
            
            if miles <= threshold:
                mileage = miles * rate1
            else:
                mileage = threshold * rate1 + (miles - threshold) * rate2
                
            # Receipt component
            if self.formula_components['receipt_rate'] is not None:
                receipt = receipts * self.formula_components['receipt_rate']
            else:
                scale = self.formula_components['receipt_scale']
                power = self.formula_components['receipt_power']
                receipt = scale * (receipts ** power)
                
            # Bonus component
            bonus = self.formula_components['average_bonus']
            
            prediction = base + mileage + receipt + bonus
            predictions.append(prediction)
            
        self.df['predicted'] = predictions
        self.df['error'] = abs(self.df['reimbursement'] - self.df['predicted'])
        
        # Calculate accuracy metrics
        mae = self.df['error'].mean()
        max_error = self.df['error'].max()
        exact_matches = len(self.df[self.df['error'] < 0.01])
        close_matches = len(self.df[self.df['error'] < 1.0])
        very_close_matches = len(self.df[self.df['error'] < 5.0])
        
        print(f"\n📊 Accuracy Results:")
        print(f"   Mean Absolute Error: ${mae:.2f}")
        print(f"   Maximum Error: ${max_error:.2f}")
        print(f"   Exact matches (±$0.01): {exact_matches}/1000 ({exact_matches/10:.1f}%)")
        print(f"   Close matches (±$1.00): {close_matches}/1000 ({close_matches/10:.1f}%)")
        print(f"   Very close matches (±$5.00): {very_close_matches}/1000 ({very_close_matches/10:.1f}%)")
        
        # Show worst cases for debugging
        worst_cases = self.df.nlargest(5, 'error')
        print(f"\n⚠️ Worst Prediction Cases:")
        for _, case in worst_cases.iterrows():
            print(f"   {case['days']}d, {case['miles']}mi, ${case['receipts']:.2f} → Expected: ${case['reimbursement']:.2f}, Got: ${case['predicted']:.2f}, Error: ${case['error']:.2f}")
            
        return {
            'mae': mae,
            'max_error': max_error,
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'exact_match_rate': exact_matches / 1000,
            'close_match_rate': close_matches / 1000
        }
        
    def generate_implementation_code(self):
        """Generate the Python implementation code"""
        print("\n💻 GENERATING IMPLEMENTATION CODE")
        print("=" * 60)
        
        code = f'''
def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Legacy reimbursement calculation discovered through reverse engineering
    """
    days = trip_duration_days
    miles = miles_traveled
    receipts = total_receipts_amount
    
    # Base per-diem component
    base = days * {self.formula_components['base_per_diem_rate']:.6f}
    
    # Mileage component (tiered)
    threshold = {self.formula_components['mileage_threshold']}
    rate1 = {self.formula_components['mileage_rate1']:.6f}
    rate2 = {self.formula_components['mileage_rate2']:.6f}
    
    if miles <= threshold:
        mileage = miles * rate1
    else:
        mileage = threshold * rate1 + (miles - threshold) * rate2
    
    # Receipt component
'''
        
        if self.formula_components['receipt_rate'] is not None:
            code += f'''    receipt = receipts * {self.formula_components['receipt_rate']:.6f}
'''
        else:
            code += f'''    scale = {self.formula_components['receipt_scale']:.6f}
    power = {self.formula_components['receipt_power']:.6f}
    receipt = scale * (receipts ** power)
'''
        
        code += f'''    
    # Bonus/adjustment component
    bonus = {self.formula_components['average_bonus']:.6f}
    
    # Final calculation
    reimbursement = base + mileage + receipt + bonus
    
    return round(reimbursement, 2)
'''
        
        print("✅ Implementation code generated!")
        
        # Save to file
        with open('src/calculator.py', 'w') as f:
            f.write(code)
            
        print("💾 Code saved to src/calculator.py")
        
        return code
        
    def run_complete_discovery(self):
        """Run the complete formula discovery process"""
        print("🚀 COMPREHENSIVE LEGACY SYSTEM FORMULA DISCOVERY")
        print("=" * 70)
        
        # Discover formula structure
        structure_results = self.discover_base_formula_structure()
        
        print(f"\n🎉 DISCOVERY PHASE 2 INITIAL COMPLETE!")
        print(f"📊 Linear model MAE: ${structure_results['linear_mae']:.2f}")
        
        return structure_results

if __name__ == "__main__":
    discovery = ComprehensiveFormulaDiscovery()
    discovery.run_complete_discovery() 