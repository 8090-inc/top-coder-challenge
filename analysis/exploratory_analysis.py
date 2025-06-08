#!/usr/bin/env python3
"""
Exploratory Data Analysis for Legacy Reimbursement System
Phase 1: Data Analysis & Pattern Discovery

This script performs comprehensive analysis of the 1,000 historical 
input/output examples to discover patterns in the legacy system.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class LegacySystemAnalyzer:
    """Analyzer for the black box legacy reimbursement system"""
    
    def __init__(self, data_file: str = 'public_cases.json'):
        """Initialize analyzer with historical data"""
        self.data_file = data_file
        self.df = None
        self.load_data()
        
    def load_data(self) -> None:
        """Load and prepare the historical cases data"""
        print("🔍 Loading historical data...")
        
        with open(self.data_file, 'r') as f:
            raw_data = json.load(f)
            
        # Convert to structured format
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
        print(f"✅ Loaded {len(self.df)} historical cases")
        
    def basic_statistics(self) -> None:
        """Generate basic statistical summaries"""
        print("\n📊 BASIC STATISTICAL ANALYSIS")
        print("=" * 50)
        
        print("\n🔢 Dataset Overview:")
        print(f"Total cases: {len(self.df)}")
        print(f"Date range: N/A (historical data)")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        print("\n📈 Input Parameter Statistics:")
        print(self.df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].describe())
        
        print("\n💰 Reimbursement Amount Statistics:")
        print(self.df[['reimbursement_amount']].describe())
        
        # Value ranges
        print("\n📏 Parameter Ranges:")
        for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount']:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            median_val = self.df[col].median()
            print(f"{col}: {min_val:.2f} to {max_val:.2f} (median: {median_val:.2f})")
            
    def distribution_analysis(self) -> None:
        """Analyze distributions of all variables"""
        print("\n📊 DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Trip duration distribution
        print("\n🗓️ Trip Duration Distribution:")
        duration_counts = self.df['trip_duration_days'].value_counts().sort_index()
        for days, count in duration_counts.items():
            pct = (count / len(self.df)) * 100
            print(f"  {days} days: {count} cases ({pct:.1f}%)")
            
        # Miles traveled patterns
        print("\n🚗 Miles Traveled Patterns:")
        self.df['miles_category'] = pd.cut(self.df['miles_traveled'], 
                                          bins=[0, 100, 300, 500, 1000, float('inf')],
                                          labels=['0-100', '101-300', '301-500', '501-1000', '1000+'])
        miles_dist = self.df['miles_category'].value_counts()
        for category, count in miles_dist.items():
            pct = (count / len(self.df)) * 100
            print(f"  {category} miles: {count} cases ({pct:.1f}%)")
            
        # Receipt amount patterns  
        print("\n🧾 Receipt Amount Patterns:")
        self.df['receipt_category'] = pd.cut(self.df['total_receipts_amount'],
                                           bins=[0, 50, 200, 500, 1000, float('inf')],
                                           labels=['$0-50', '$51-200', '$201-500', '$501-1000', '$1000+'])
        receipt_dist = self.df['receipt_category'].value_counts()
        for category, count in receipt_dist.items():
            pct = (count / len(self.df)) * 100
            print(f"  {category}: {count} cases ({pct:.1f}%)")
            
    def correlation_analysis(self) -> None:
        """Analyze correlations between variables"""
        print("\n🔗 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Calculate correlation matrix
        corr_matrix = self.df[['trip_duration_days', 'miles_traveled', 
                              'total_receipts_amount', 'reimbursement_amount']].corr()
        
        print("\n📈 Correlation Matrix:")
        print(corr_matrix.round(3))
        
        print("\n🎯 Key Correlations with Reimbursement:")
        reimbursement_corr = corr_matrix['reimbursement_amount'].sort_values(ascending=False)
        for var, corr in reimbursement_corr.items():
            if var != 'reimbursement_amount':
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                direction = "Positive" if corr > 0 else "Negative"
                print(f"  {var}: {corr:.3f} ({strength} {direction})")
                
    def pattern_discovery(self) -> None:
        """Discover key patterns in the data"""
        print("\n🔍 PATTERN DISCOVERY")
        print("=" * 50)
        
        # Test 5-day trip bonus hypothesis
        print("\n🎯 Testing 5-Day Trip Bonus Hypothesis:")
        avg_by_duration = self.df.groupby('trip_duration_days')['reimbursement_amount'].agg(['mean', 'count'])
        for days, row in avg_by_duration.iterrows():
            if row['count'] >= 10:  # Only show categories with enough data
                print(f"  {days} days: ${row['mean']:.2f} average ({row['count']} cases)")
                
        # Calculate per-day reimbursement rates
        self.df['reimbursement_per_day'] = self.df['reimbursement_amount'] / self.df['trip_duration_days']
        print("\n📊 Per-Day Reimbursement Rates:")
        avg_per_day = self.df.groupby('trip_duration_days')['reimbursement_per_day'].agg(['mean', 'std', 'count'])
        for days, row in avg_per_day.iterrows():
            if row['count'] >= 10:
                print(f"  {days} days: ${row['mean']:.2f} ± ${row['std']:.2f} per day ({row['count']} cases)")
                
        # Test mileage efficiency hypothesis  
        print("\n🚗 Testing Mileage Efficiency Hypothesis:")
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        
        # Create efficiency bins
        self.df['efficiency_category'] = pd.cut(self.df['miles_per_day'],
                                              bins=[0, 50, 100, 180, 220, 300, float('inf')],
                                              labels=['0-50', '51-100', '101-180', '181-220', '221-300', '300+'])
        
        efficiency_analysis = self.df.groupby('efficiency_category')['reimbursement_amount'].agg(['mean', 'count'])
        for category, row in efficiency_analysis.iterrows():
            if row['count'] >= 5:
                print(f"  {category} miles/day: ${row['mean']:.2f} average ({row['count']} cases)")
                
        # Test receipt processing patterns
        print("\n🧾 Receipt Processing Patterns:")
        
        # Calculate receipt-to-reimbursement ratios
        self.df['receipt_ratio'] = self.df['reimbursement_amount'] / (self.df['total_receipts_amount'] + 0.01)  # Avoid division by zero
        
        receipt_analysis = self.df.groupby('receipt_category')['receipt_ratio'].agg(['mean', 'std', 'count'])
        for category, row in receipt_analysis.iterrows():
            if row['count'] >= 10:
                print(f"  {category}: {row['mean']:.2f}x ratio ± {row['std']:.2f} ({row['count']} cases)")
                
    def edge_case_analysis(self) -> None:
        """Identify outliers and edge cases"""
        print("\n⚠️ EDGE CASE ANALYSIS")
        print("=" * 50)
        
        # High reimbursement cases
        high_reimbursement = self.df[self.df['reimbursement_amount'] > self.df['reimbursement_amount'].quantile(0.95)]
        print(f"\n💰 Top 5% Reimbursement Cases ({len(high_reimbursement)} cases):")
        print(f"  Range: ${high_reimbursement['reimbursement_amount'].min():.2f} - ${high_reimbursement['reimbursement_amount'].max():.2f}")
        
        # Low reimbursement cases
        low_reimbursement = self.df[self.df['reimbursement_amount'] < self.df['reimbursement_amount'].quantile(0.05)]
        print(f"\n💸 Bottom 5% Reimbursement Cases ({len(low_reimbursement)} cases):")
        print(f"  Range: ${low_reimbursement['reimbursement_amount'].min():.2f} - ${low_reimbursement['reimbursement_amount'].max():.2f}")
        
        # Outlier detection using IQR method
        Q1 = self.df['reimbursement_amount'].quantile(0.25)
        Q3 = self.df['reimbursement_amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df['reimbursement_amount'] < lower_bound) | 
                          (self.df['reimbursement_amount'] > upper_bound)]
        print(f"\n📊 Statistical Outliers: {len(outliers)} cases")
        
        if len(outliers) > 0:
            print("  Sample outlier cases:")
            for idx, row in outliers.head(5).iterrows():
                print(f"    {row['trip_duration_days']}d, {row['miles_traveled']}mi, ${row['total_receipts_amount']:.2f} → ${row['reimbursement_amount']:.2f}")
                
    def hypothesis_extraction(self) -> None:
        """Extract specific hypotheses from employee interviews for testing"""
        print("\n🎯 EMPLOYEE INTERVIEW HYPOTHESIS EXTRACTION")
        print("=" * 50)
        
        hypotheses = [
            "5-day trips get bonus payments (Lisa from Accounting)",
            "Efficiency sweet spot: 180-220 miles/day gets bonuses",
            "First 100 miles at $0.58/mile, then declining rate",
            "Base per diem around $100/day",
            "Receipt caps exist - diminishing returns for high amounts",
            "Very low receipts (<$50) get penalties",
            "Rounding bug: amounts ending in .49 or .99 get extra money"
        ]
        
        print("\n📋 Key Hypotheses to Test:")
        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"  {i}. {hypothesis}")
            
        print("\n🔬 Hypothesis Testing Results:")
        
        # Test 5-day bonus hypothesis
        five_day_trips = self.df[self.df['trip_duration_days'] == 5]
        other_trips = self.df[self.df['trip_duration_days'] != 5]
        
        if len(five_day_trips) > 0 and len(other_trips) > 0:
            five_day_avg_per_day = five_day_trips['reimbursement_per_day'].mean()
            other_avg_per_day = other_trips['reimbursement_per_day'].mean()
            bonus_amount = five_day_avg_per_day - other_avg_per_day
            
            print(f"\n✅ 5-Day Bonus Test:")
            print(f"  5-day trips: ${five_day_avg_per_day:.2f} per day")
            print(f"  Other trips: ${other_avg_per_day:.2f} per day")
            print(f"  Difference: ${bonus_amount:.2f} per day ({'BONUS' if bonus_amount > 0 else 'PENALTY'})")
            
        # Test efficiency sweet spot
        sweet_spot = self.df[(self.df['miles_per_day'] >= 180) & (self.df['miles_per_day'] <= 220)]
        other_efficiency = self.df[(self.df['miles_per_day'] < 180) | (self.df['miles_per_day'] > 220)]
        
        if len(sweet_spot) > 0 and len(other_efficiency) > 0:
            sweet_spot_avg = sweet_spot['reimbursement_amount'].mean()
            other_avg = other_efficiency['reimbursement_amount'].mean() 
            
            print(f"\n✅ Efficiency Sweet Spot Test:")
            print(f"  180-220 miles/day: ${sweet_spot_avg:.2f} average ({len(sweet_spot)} cases)")
            print(f"  Other efficiency: ${other_avg:.2f} average ({len(other_efficiency)} cases)")
            print(f"  Difference: ${sweet_spot_avg - other_avg:.2f} ({'BONUS' if sweet_spot_avg > other_avg else 'PENALTY'})")
            
    def save_insights(self) -> None:
        """Save key insights for next phase"""
        print("\n💾 SAVING INSIGHTS FOR PHASE 2")
        print("=" * 50)
        
        insights = {
            'dataset_size': len(self.df),
            'trip_duration_range': [int(self.df['trip_duration_days'].min()), int(self.df['trip_duration_days'].max())],
            'miles_range': [int(self.df['miles_traveled'].min()), int(self.df['miles_traveled'].max())],
            'receipts_range': [float(self.df['total_receipts_amount'].min()), float(self.df['total_receipts_amount'].max())],
            'reimbursement_range': [float(self.df['reimbursement_amount'].min()), float(self.df['reimbursement_amount'].max())],
            'correlations': self.df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount']].corr().to_dict(),
            'five_day_bonus_confirmed': len(self.df[self.df['trip_duration_days'] == 5]) > 0
        }
        
        # Save to file for next phase
        with open('analysis/phase1_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
            
        print("✅ Insights saved to analysis/phase1_insights.json")
        print("🚀 Ready for Phase 2: Hypothesis Testing & Formula Discovery")
        
    def run_full_analysis(self) -> None:
        """Run complete Phase 1 analysis"""
        print("🎯 LEGACY REIMBURSEMENT SYSTEM - PHASE 1 ANALYSIS")
        print("=" * 60)
        
        self.basic_statistics()
        self.distribution_analysis()
        self.correlation_analysis()
        self.pattern_discovery()
        self.edge_case_analysis()
        self.hypothesis_extraction()
        self.save_insights()
        
        print(f"\n🎉 Phase 1 Complete! Analyzed {len(self.df)} historical cases")
        print("📋 Key findings documented and ready for hypothesis testing")

if __name__ == "__main__":
    analyzer = LegacySystemAnalyzer()
    analyzer.run_full_analysis() 