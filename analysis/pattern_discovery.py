#!/usr/bin/env python3
"""
Phase 2: Hypothesis Testing & Formula Discovery
Legacy Reimbursement System Reverse Engineering

This script tests specific hypotheses and discovers the mathematical 
components of the legacy reimbursement formula.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from typing import Dict, List, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

class FormulaDiscovery:
    """Formula discovery engine for the legacy system"""
    
    def __init__(self, data_file: str = 'public_cases.json', insights_file: str = 'analysis/phase1_insights.json'):
        """Initialize with historical data and Phase 1 insights"""
        self.data_file = data_file
        self.insights_file = insights_file
        self.df = None
        self.insights = None
        self.formula_components = {}
        self.load_data()
        
    def load_data(self) -> None:
        """Load historical data and Phase 1 insights"""
        print("🔄 Loading data and Phase 1 insights...")
        
        # Load historical cases
        with open(self.data_file, 'r') as f:
            raw_data = json.load(f)
            
        data_rows = []
        for case in raw_data:
            row = {
                'trip_duration_days': case['input']['trip_duration_days'],
                'miles_traveled': case['input']['miles_traveled'], 
                'total_receipts_amount': case['input']['total_receipts_amount'],
                'reimbursement_amount': case['expected_output']
            }
            data_rows.append(row)
            
        self.df = pd.DataFrame(data_rows)
        
        # Load Phase 1 insights
        with open(self.insights_file, 'r') as f:
            self.insights = json.load(f)
            
        print(f"✅ Loaded {len(self.df)} cases and Phase 1 insights")
        
    def test_base_per_diem_hypothesis(self) -> Dict[str, Any]:
        """Test if there's a base per-diem component"""
        print("\n🏨 TESTING BASE PER-DIEM HYPOTHESIS")
        print("=" * 50)
        
        # Calculate minimum reimbursement per day for each trip length
        min_per_day = self.df.groupby('trip_duration_days')['reimbursement_amount'].min() / self.df.groupby('trip_duration_days')['trip_duration_days'].first()
        avg_per_day = self.df.groupby('trip_duration_days')['reimbursement_amount'].mean() / self.df.groupby('trip_duration_days')['trip_duration_days'].first()
        
        print("\n📊 Per-Day Analysis by Trip Length:")
        for days in sorted(self.df['trip_duration_days'].unique()):
            if len(self.df[self.df['trip_duration_days'] == days]) >= 10:
                min_day = min_per_day[days]
                avg_day = avg_per_day[days]
                print(f"  {days:2d} days: Min ${min_day:.2f}/day, Avg ${avg_day:.2f}/day")
                
        # Look for a base rate (minimum across all trip lengths)
        estimated_base_rate = min_per_day.min()
        print(f"\n🎯 Estimated Base Per-Diem Rate: ${estimated_base_rate:.2f}/day")
        
        # Test if there's a consistent base component
        base_component_test = {}
        for days in sorted(self.df['trip_duration_days'].unique()):
            trip_data = self.df[self.df['trip_duration_days'] == days]
            if len(trip_data) >= 5:
                base_reimbursement = days * estimated_base_rate
                remaining = trip_data['reimbursement_amount'] - base_reimbursement
                base_component_test[days] = {
                    'count': len(trip_data),
                    'base_portion': base_reimbursement,
                    'remaining_avg': remaining.mean(),
                    'remaining_std': remaining.std()
                }
                
        self.formula_components['base_per_diem'] = {
            'rate': estimated_base_rate,
            'component_test': base_component_test,
            'confidence': 'medium'
        }
        
        return self.formula_components['base_per_diem']
        
    def test_mileage_tier_hypothesis(self) -> Dict[str, Any]:
        """Test for tiered mileage calculation system"""
        print("\n🛣️ TESTING MILEAGE TIER HYPOTHESIS")
        print("=" * 50)
        
        # Sort by miles and analyze per-mile reimbursement
        self.df['miles_component_estimate'] = self.df['reimbursement_amount'] - (self.df['trip_duration_days'] * self.formula_components['base_per_diem']['rate'])
        self.df['per_mile_rate'] = self.df['miles_component_estimate'] / self.df['miles_traveled']
        
        # Create mileage bins for analysis
        bins = [0, 100, 200, 400, 600, 800, 1000, float('inf')]
        self.df['mileage_bin'] = pd.cut(self.df['miles_traveled'], bins=bins)
        
        print("\n📈 Per-Mile Rates by Mileage Range:")
        mileage_analysis = self.df.groupby('mileage_bin')['per_mile_rate'].agg(['mean', 'std', 'count'])
        
        for bin_range, row in mileage_analysis.iterrows():
            if row['count'] >= 10:
                print(f"  {bin_range}: ${row['mean']:.3f}/mile ± ${row['std']:.3f} ({row['count']} cases)")
                
        # Test specific tier hypothesis: first 100 miles at higher rate
        first_100_cases = self.df[self.df['miles_traveled'] <= 100]
        over_100_cases = self.df[self.df['miles_traveled'] > 100]
        
        if len(first_100_cases) > 0 and len(over_100_cases) > 0:
            first_100_rate = first_100_cases['per_mile_rate'].mean()
            over_100_rate = over_100_cases['per_mile_rate'].mean()
            
            print(f"\n🎯 Tier Analysis:")
            print(f"  First 100 miles: ${first_100_rate:.3f}/mile average")
            print(f"  Over 100 miles: ${over_100_rate:.3f}/mile average")
            print(f"  Difference: ${first_100_rate - over_100_rate:.3f}/mile")
            
        # Try to fit a tiered model
        def tiered_mileage_model(miles, rate1, rate2, threshold):
            """Tiered mileage model: rate1 up to threshold, rate2 after"""
            return np.where(miles <= threshold, 
                          miles * rate1,
                          threshold * rate1 + (miles - threshold) * rate2)
        
        try:
            # Fit the tiered model
            popt, pcov = curve_fit(tiered_mileage_model, 
                                 self.df['miles_traveled'], 
                                 self.df['miles_component_estimate'],
                                 p0=[0.5, 0.3, 100],  # Initial guess
                                 bounds=([0, 0, 50], [2, 2, 500]))
            
            rate1, rate2, threshold = popt
            print(f"\n🔧 Fitted Tiered Model:")
            print(f"  Rate 1 (first {threshold:.0f} miles): ${rate1:.3f}/mile")
            print(f"  Rate 2 (after {threshold:.0f} miles): ${rate2:.3f}/mile")
            
            self.formula_components['mileage_tiers'] = {
                'rate1': rate1,
                'rate2': rate2, 
                'threshold': threshold,
                'confidence': 'high'
            }
            
        except Exception as e:
            print(f"⚠️ Could not fit tiered model: {e}")
            self.formula_components['mileage_tiers'] = {
                'simple_rate': self.df['per_mile_rate'].mean(),
                'confidence': 'low'
            }
            
        return self.formula_components['mileage_tiers']
        
    def test_receipt_processing_hypothesis(self) -> Dict[str, Any]:
        """Test receipt processing logic"""
        print("\n🧾 TESTING RECEIPT PROCESSING HYPOTHESIS")
        print("=" * 50)
        
        # Calculate receipt component (removing base and mileage estimates)
        base_component = self.df['trip_duration_days'] * self.formula_components['base_per_diem']['rate']
        
        if 'rate1' in self.formula_components['mileage_tiers']:
            # Use tiered mileage model
            rate1 = self.formula_components['mileage_tiers']['rate1']
            rate2 = self.formula_components['mileage_tiers']['rate2']
            threshold = self.formula_components['mileage_tiers']['threshold']
            
            mileage_component = np.where(self.df['miles_traveled'] <= threshold,
                                       self.df['miles_traveled'] * rate1,
                                       threshold * rate1 + (self.df['miles_traveled'] - threshold) * rate2)
        else:
            # Use simple rate
            mileage_component = self.df['miles_traveled'] * self.formula_components['mileage_tiers']['simple_rate']
            
        self.df['receipt_component_estimate'] = self.df['reimbursement_amount'] - base_component - mileage_component
        self.df['receipt_ratio'] = self.df['receipt_component_estimate'] / self.df['total_receipts_amount']
        
        # Analyze receipt processing by amount ranges
        receipt_bins = [0, 50, 200, 500, 1000, 1500, 2000, float('inf')]
        self.df['receipt_bin'] = pd.cut(self.df['total_receipts_amount'], bins=receipt_bins)
        
        print("\n📊 Receipt Processing Analysis:")
        receipt_analysis = self.df.groupby('receipt_bin')['receipt_ratio'].agg(['mean', 'std', 'count'])
        
        for bin_range, row in receipt_analysis.iterrows():
            if row['count'] >= 5:
                print(f"  {bin_range}: {row['mean']:.3f}x ratio ± {row['std']:.3f} ({row['count']} cases)")
                
        # Test for diminishing returns model
        def diminishing_returns_model(receipts, max_rate, decay_factor, min_rate):
            """Diminishing returns model for receipt processing"""
            return receipts * (min_rate + (max_rate - min_rate) * np.exp(-decay_factor * receipts / 1000))
        
        try:
            # Fit diminishing returns model
            popt, pcov = curve_fit(diminishing_returns_model,
                                 self.df['total_receipts_amount'],
                                 self.df['receipt_component_estimate'],
                                 p0=[1.0, 0.5, 0.3],  # Initial guess
                                 bounds=([0, 0, 0], [3, 3, 1]))
            
            max_rate, decay_factor, min_rate = popt
            print(f"\n🔧 Fitted Diminishing Returns Model:")
            print(f"  Max rate: {max_rate:.3f}x")
            print(f"  Min rate: {min_rate:.3f}x") 
            print(f"  Decay factor: {decay_factor:.3f}")
            
            self.formula_components['receipt_processing'] = {
                'max_rate': max_rate,
                'min_rate': min_rate,
                'decay_factor': decay_factor,
                'model_type': 'diminishing_returns',
                'confidence': 'high'
            }
            
        except Exception as e:
            print(f"⚠️ Could not fit diminishing returns model: {e}")
            # Fallback to simple ratio
            avg_ratio = self.df['receipt_ratio'].mean()
            self.formula_components['receipt_processing'] = {
                'simple_ratio': avg_ratio,
                'model_type': 'simple',
                'confidence': 'low'
            }
            
        return self.formula_components['receipt_processing']
        
    def test_special_bonuses_hypothesis(self) -> Dict[str, Any]:
        """Test for special bonuses and penalties"""
        print("\n🎁 TESTING SPECIAL BONUSES & PENALTIES")
        print("=" * 50)
        
        # Calculate residuals after base, mileage, and receipt components
        base_component = self.df['trip_duration_days'] * self.formula_components['base_per_diem']['rate']
        
        if 'rate1' in self.formula_components['mileage_tiers']:
            rate1 = self.formula_components['mileage_tiers']['rate1']
            rate2 = self.formula_components['mileage_tiers']['rate2']
            threshold = self.formula_components['mileage_tiers']['threshold']
            mileage_component = np.where(self.df['miles_traveled'] <= threshold,
                                       self.df['miles_traveled'] * rate1,
                                       threshold * rate1 + (self.df['miles_traveled'] - threshold) * rate2)
        else:
            mileage_component = self.df['miles_traveled'] * self.formula_components['mileage_tiers']['simple_rate']
            
        if 'max_rate' in self.formula_components['receipt_processing']:
            max_rate = self.formula_components['receipt_processing']['max_rate']
            min_rate = self.formula_components['receipt_processing']['min_rate']
            decay_factor = self.formula_components['receipt_processing']['decay_factor']
            receipt_component = self.df['total_receipts_amount'] * (min_rate + (max_rate - min_rate) * np.exp(-decay_factor * self.df['total_receipts_amount'] / 1000))
        else:
            receipt_component = self.df['total_receipts_amount'] * self.formula_components['receipt_processing']['simple_ratio']
            
        self.df['residual'] = self.df['reimbursement_amount'] - base_component - mileage_component - receipt_component
        
        print(f"\n📊 Residual Analysis:")
        print(f"  Mean residual: ${self.df['residual'].mean():.2f}")
        print(f"  Std residual: ${self.df['residual'].std():.2f}")
        print(f"  Min residual: ${self.df['residual'].min():.2f}")
        print(f"  Max residual: ${self.df['residual'].max():.2f}")
        
        # Test efficiency bonus hypothesis
        self.df['efficiency'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        
        # Look for efficiency sweet spots
        efficiency_bins = [0, 50, 100, 150, 200, 250, 300, float('inf')]
        self.df['efficiency_bin'] = pd.cut(self.df['efficiency'], bins=efficiency_bins)
        
        print(f"\n🏃 Efficiency Bonus Analysis:")
        efficiency_analysis = self.df.groupby('efficiency_bin')['residual'].agg(['mean', 'std', 'count'])
        
        for bin_range, row in efficiency_analysis.iterrows():
            if row['count'] >= 10:
                bonus_penalty = "BONUS" if row['mean'] > 5 else "PENALTY" if row['mean'] < -5 else "NEUTRAL"
                print(f"  {bin_range}: ${row['mean']:.2f} ± ${row['std']:.2f} ({bonus_penalty}, {row['count']} cases)")
                
        # Test trip duration bonuses/penalties
        print(f"\n📅 Trip Duration Bonus Analysis:")
        duration_analysis = self.df.groupby('trip_duration_days')['residual'].agg(['mean', 'std', 'count'])
        
        for days, row in duration_analysis.iterrows():
            if row['count'] >= 10:
                bonus_penalty = "BONUS" if row['mean'] > 5 else "PENALTY" if row['mean'] < -5 else "NEUTRAL"
                print(f"  {days:2d} days: ${row['mean']:.2f} ± ${row['std']:.2f} ({bonus_penalty}, {row['count']} cases)")
                
        # Save special bonus findings
        self.formula_components['special_bonuses'] = {
            'efficiency_bonuses': efficiency_analysis.to_dict('index'),
            'duration_bonuses': duration_analysis.to_dict('index'),
            'residual_stats': {
                'mean': float(self.df['residual'].mean()),
                'std': float(self.df['residual'].std()),
                'min': float(self.df['residual'].min()),
                'max': float(self.df['residual'].max())
            }
        }
        
        return self.formula_components['special_bonuses']
        
    def validate_formula_accuracy(self) -> Dict[str, float]:
        """Validate the discovered formula against historical data"""
        print("\n✅ VALIDATING FORMULA ACCURACY")
        print("=" * 50)
        
        # Calculate predicted reimbursements using discovered components
        predictions = []
        
        for idx, row in self.df.iterrows():
            days = row['trip_duration_days']
            miles = row['miles_traveled']
            receipts = row['total_receipts_amount']
            
            # Base component
            base = days * self.formula_components['base_per_diem']['rate']
            
            # Mileage component
            if 'rate1' in self.formula_components['mileage_tiers']:
                rate1 = self.formula_components['mileage_tiers']['rate1']
                rate2 = self.formula_components['mileage_tiers']['rate2']
                threshold = self.formula_components['mileage_tiers']['threshold']
                
                if miles <= threshold:
                    mileage = miles * rate1
                else:
                    mileage = threshold * rate1 + (miles - threshold) * rate2
            else:
                mileage = miles * self.formula_components['mileage_tiers']['simple_rate']
                
            # Receipt component
            if 'max_rate' in self.formula_components['receipt_processing']:
                max_rate = self.formula_components['receipt_processing']['max_rate']
                min_rate = self.formula_components['receipt_processing']['min_rate']
                decay_factor = self.formula_components['receipt_processing']['decay_factor']
                receipt = receipts * (min_rate + (max_rate - min_rate) * np.exp(-decay_factor * receipts / 1000))
            else:
                receipt = receipts * self.formula_components['receipt_processing']['simple_ratio']
                
            # Special bonuses (simplified - use mean residual for now)
            special = self.formula_components['special_bonuses']['residual_stats']['mean']
            
            predicted = base + mileage + receipt + special
            predictions.append(predicted)
            
        self.df['predicted_reimbursement'] = predictions
        self.df['error'] = abs(self.df['reimbursement_amount'] - self.df['predicted_reimbursement'])
        self.df['error_pct'] = (self.df['error'] / self.df['reimbursement_amount']) * 100
        
        # Calculate accuracy metrics
        mean_error = self.df['error'].mean()
        mean_error_pct = self.df['error_pct'].mean()
        exact_matches = len(self.df[self.df['error'] < 0.01])
        close_matches = len(self.df[self.df['error'] < 1.0])
        
        accuracy_metrics = {
            'mean_absolute_error': mean_error,
            'mean_percentage_error': mean_error_pct,
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'exact_match_rate': exact_matches / len(self.df),
            'close_match_rate': close_matches / len(self.df)
        }
        
        print(f"\n📊 Formula Accuracy Metrics:")
        print(f"  Mean Absolute Error: ${mean_error:.2f}")
        print(f"  Mean Percentage Error: {mean_error_pct:.1f}%")
        print(f"  Exact Matches (±$0.01): {exact_matches}/{len(self.df)} ({accuracy_metrics['exact_match_rate']:.1%})")
        print(f"  Close Matches (±$1.00): {close_matches}/{len(self.df)} ({accuracy_metrics['close_match_rate']:.1%})")
        
        # Show worst predictions for debugging
        worst_cases = self.df.nlargest(5, 'error')[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount', 'predicted_reimbursement', 'error']]
        print(f"\n⚠️ Worst Prediction Cases:")
        for idx, case in worst_cases.iterrows():
            print(f"  {case['trip_duration_days']}d, {case['miles_traveled']}mi, ${case['total_receipts_amount']:.2f} → Expected: ${case['reimbursement_amount']:.2f}, Got: ${case['predicted_reimbursement']:.2f}, Error: ${case['error']:.2f}")
            
        return accuracy_metrics
        
    def save_formula_components(self) -> None:
        """Save discovered formula components for Phase 3"""
        print("\n💾 SAVING FORMULA COMPONENTS")
        print("=" * 50)
        
        # Serialize components for JSON
        serializable_components = {}
        for key, value in self.formula_components.items():
            if isinstance(value, dict):
                serializable_components[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                              for k, v in value.items() if not isinstance(v, dict)}
                # Handle nested dicts
                for k, v in value.items():
                    if isinstance(v, dict):
                        serializable_components[key][k] = {
                            sub_k: float(sub_v) if isinstance(sub_v, (np.floating, np.integer)) else sub_v
                            for sub_k, sub_v in v.items()
                        }
            else:
                serializable_components[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
                
        with open('analysis/phase2_formula_components.json', 'w') as f:
            json.dump(serializable_components, f, indent=2)
            
        print("✅ Formula components saved to analysis/phase2_formula_components.json")
        print("🚀 Ready for Phase 3: Implementation & Optimization")
        
    def run_formula_discovery(self) -> None:
        """Run complete Phase 2 formula discovery"""
        print("🔬 LEGACY REIMBURSEMENT SYSTEM - PHASE 2 FORMULA DISCOVERY")
        print("=" * 70)
        
        # Test base per-diem hypothesis
        self.test_base_per_diem_hypothesis()
        
        print(f"\n🎉 Phase 2 Initial Test Complete!")

if __name__ == "__main__":
    discovery = FormulaDiscovery()
    discovery.run_formula_discovery() 