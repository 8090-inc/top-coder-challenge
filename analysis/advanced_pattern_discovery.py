#!/usr/bin/env python3
"""
Advanced Pattern Discovery for Legacy Reimbursement System
Phase 2B: Non-Linear Formula Discovery

This script looks for caps, thresholds, and non-linear relationships
based on the poor performance of the linear model.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedPatternDiscovery:
    """Advanced pattern discovery for non-linear relationships"""
    
    def __init__(self):
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the historical data"""
        print("🔍 Loading data for advanced pattern analysis...")
        
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
        
    def analyze_receipt_caps_and_thresholds(self):
        """Look for caps and thresholds in receipt processing"""
        print("\n🧾 ANALYZING RECEIPT CAPS & THRESHOLDS")
        print("=" * 60)
        
        # Sort by receipts and look for patterns
        df_sorted = self.df.sort_values('receipts')
        
        # Calculate per-receipt reimbursement rate
        df_sorted['per_receipt_rate'] = df_sorted['reimbursement'] / df_sorted['receipts']
        
        print("\n📊 Per-Receipt Rate Analysis:")
        
        # Create receipt buckets for analysis
        receipt_bins = [0, 50, 200, 500, 1000, 1500, 2000, 2500, float('inf')]
        df_sorted['receipt_bin'] = pd.cut(df_sorted['receipts'], bins=receipt_bins)
        
        receipt_analysis = df_sorted.groupby('receipt_bin').agg({
            'per_receipt_rate': ['mean', 'std'],
            'reimbursement': ['mean', 'std'],
            'receipts': ['mean', 'count']
        }).round(3)
        
        print(receipt_analysis)
        
        # Look for evidence of caps
        high_receipt_cases = self.df[self.df['receipts'] > 1500]
        if len(high_receipt_cases) > 0:
            print(f"\n🚨 High Receipt Cases (>${1500}):")
            print(f"   Average reimbursement: ${high_receipt_cases['reimbursement'].mean():.2f}")
            print(f"   Average per-receipt rate: {high_receipt_cases['reimbursement'].sum() / high_receipt_cases['receipts'].sum():.3f}")
            
        # Compare with low/medium receipt cases
        low_receipt_cases = self.df[self.df['receipts'] <= 500]
        if len(low_receipt_cases) > 0:
            print(f"\n💰 Low Receipt Cases (<=${500}):")
            print(f"   Average reimbursement: ${low_receipt_cases['reimbursement'].mean():.2f}")
            print(f"   Average per-receipt rate: {low_receipt_cases['reimbursement'].sum() / low_receipt_cases['receipts'].sum():.3f}")
            
    def analyze_mileage_caps_and_thresholds(self):
        """Look for caps and thresholds in mileage processing"""
        print("\n🛣️ ANALYZING MILEAGE CAPS & THRESHOLDS")
        print("=" * 60)
        
        # Sort by miles and look for patterns
        df_sorted = self.df.sort_values('miles')
        
        # Calculate per-mile reimbursement rate (accounting for other factors)
        # Simple approximation: subtract a base amount and divide by miles
        base_estimate = 200  # Rough estimate for base reimbursement
        df_sorted['adjusted_reimbursement'] = df_sorted['reimbursement'] - base_estimate
        df_sorted['per_mile_rate'] = df_sorted['adjusted_reimbursement'] / df_sorted['miles']
        
        print("\n📊 Per-Mile Rate Analysis:")
        
        # Create mileage buckets
        mile_bins = [0, 100, 300, 500, 700, 1000, float('inf')]
        df_sorted['mile_bin'] = pd.cut(df_sorted['miles'], bins=mile_bins)
        
        mile_analysis = df_sorted.groupby('mile_bin').agg({
            'per_mile_rate': ['mean', 'std'],
            'reimbursement': ['mean', 'std'],
            'miles': ['mean', 'count']
        }).round(3)
        
        print(mile_analysis)
        
        # Look for diminishing returns pattern
        high_mile_cases = self.df[self.df['miles'] > 800]
        low_mile_cases = self.df[self.df['miles'] <= 200]
        
        if len(high_mile_cases) > 0 and len(low_mile_cases) > 0:
            high_avg = (high_mile_cases['reimbursement'] - base_estimate).sum() / high_mile_cases['miles'].sum()
            low_avg = (low_mile_cases['reimbursement'] - base_estimate).sum() / low_mile_cases['miles'].sum()
            
            print(f"\n📈 Mileage Rate Comparison:")
            print(f"   High mileage (>800): ${high_avg:.3f}/mile")
            print(f"   Low mileage (≤200): ${low_avg:.3f}/mile")
            print(f"   Ratio: {low_avg/high_avg:.2f}x (higher = diminishing returns)")
            
    def discover_tree_based_patterns(self):
        """Use decision trees to discover hidden patterns"""
        print("\n🌳 DECISION TREE PATTERN DISCOVERY")
        print("=" * 60)
        
        # Prepare features
        X = self.df[['days', 'miles', 'receipts']]
        y = self.df['reimbursement']
        
        # Train decision tree
        dt = DecisionTreeRegressor(max_depth=8, min_samples_split=20, random_state=42)
        dt.fit(X, y)
        
        # Calculate accuracy
        y_pred = dt.predict(X)
        mae = mean_absolute_error(y, y_pred)
        
        print(f"📊 Decision Tree Results:")
        print(f"   MAE: ${mae:.2f}")
        print(f"   Feature Importances:")
        for feature, importance in zip(['days', 'miles', 'receipts'], dt.feature_importances_):
            print(f"     {feature}: {importance:.3f}")
            
        # Extract some decision rules
        print(f"\n🔍 Sample Decision Rules (first few splits):")
        self.print_tree_rules(dt, X.columns, max_depth=3)
        
        return dt, mae
        
    def print_tree_rules(self, tree, feature_names, max_depth=3):
        """Print human-readable decision tree rules"""
        def recurse(node, depth, parent_rule=""):
            if depth > max_depth:
                return
                
            if tree.tree_.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.tree_.feature[node]]
                threshold = tree.tree_.threshold[node]
                
                left_rule = f"{parent_rule} {feature} ≤ {threshold:.2f}"
                right_rule = f"{parent_rule} {feature} > {threshold:.2f}"
                
                if depth < max_depth:
                    print(f"{'  ' * depth}├─ If {feature} ≤ {threshold:.2f}")
                    recurse(tree.tree_.children_left[node], depth + 1, left_rule)
                    
                    print(f"{'  ' * depth}└─ If {feature} > {threshold:.2f}")
                    recurse(tree.tree_.children_right[node], depth + 1, right_rule)
            else:
                value = tree.tree_.value[node][0][0]
                print(f"{'  ' * depth}→ Predict: ${value:.2f}")
                
        recurse(0, 0)
        
    def test_component_based_model_v2(self):
        """Test an improved component-based model"""
        print("\n🔧 TESTING IMPROVED COMPONENT MODEL")
        print("=" * 60)
        
        # Hypothesis: Different calculation for different trip categories
        
        # Category 1: Short trips (1-3 days)
        short_trips = self.df[self.df['days'] <= 3]
        
        # Category 2: Medium trips (4-7 days)  
        medium_trips = self.df[(self.df['days'] >= 4) & (self.df['days'] <= 7)]
        
        # Category 3: Long trips (8+ days)
        long_trips = self.df[self.df['days'] >= 8]
        
        print(f"📊 Trip Categories:")
        print(f"   Short trips (1-3 days): {len(short_trips)} cases")
        print(f"   Medium trips (4-7 days): {len(medium_trips)} cases")
        print(f"   Long trips (8+ days): {len(long_trips)} cases")
        
        # Analyze each category separately
        for name, category_df in [("Short", short_trips), ("Medium", medium_trips), ("Long", long_trips)]:
            if len(category_df) > 10:
                print(f"\n🎯 {name} Trips Analysis:")
                
                # Simple per-day rate
                avg_per_day = (category_df['reimbursement'] / category_df['days']).mean()
                print(f"   Average per-day: ${avg_per_day:.2f}")
                
                # Correlation with other factors
                mile_corr = category_df[['miles', 'reimbursement']].corr().iloc[0, 1]
                receipt_corr = category_df[['receipts', 'reimbursement']].corr().iloc[0, 1]
                
                print(f"   Miles correlation: {mile_corr:.3f}")
                print(f"   Receipts correlation: {receipt_corr:.3f}")
                
                # Look for simple patterns
                avg_reimbursement = category_df['reimbursement'].mean()
                avg_receipts = category_df['receipts'].mean()
                receipt_ratio = avg_reimbursement / avg_receipts if avg_receipts > 0 else 0
                
                print(f"   Avg reimbursement to receipts ratio: {receipt_ratio:.3f}")
                
    def run_advanced_analysis(self):
        """Run the complete advanced pattern analysis"""
        print("🔬 ADVANCED PATTERN DISCOVERY FOR LEGACY SYSTEM")
        print("=" * 70)
        
        # Analyze receipt patterns
        self.analyze_receipt_caps_and_thresholds()
        
        # Analyze mileage patterns
        self.analyze_mileage_caps_and_thresholds()
        
        # Use machine learning to find patterns
        dt, mae = self.discover_tree_based_patterns()
        
        # Test component-based approach
        self.test_component_based_model_v2()
        
        print(f"\n🎉 ADVANCED ANALYSIS COMPLETE!")
        print(f"🌳 Decision tree MAE: ${mae:.2f}")
        
        if mae < 50:
            print("🏆 EXCELLENT! Decision tree found strong patterns!")
        elif mae < 100:
            print("🥈 GOOD! Patterns found, needs refinement")
        else:
            print("🔧 NEEDS WORK! Complex patterns require deeper analysis")
            
        return dt, mae

if __name__ == "__main__":
    discovery = AdvancedPatternDiscovery()
    discovery.run_advanced_analysis() 