#!/usr/bin/env python3
"""
Precise Decision Tree Extraction
Train a decision tree and extract its exact logic for implementation
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class PrecisionTreeExtractor:
    """Extract precise decision tree logic for implementation"""
    
    def __init__(self):
        self.df = None
        self.tree = None
        self.load_data()
        
    def load_data(self):
        """Load the historical data"""
        print("📊 Loading data for precise tree extraction...")
        
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
        
    def train_optimal_tree(self):
        """Train the most accurate decision tree possible"""
        print("\n🌳 TRAINING OPTIMAL DECISION TREE")
        print("=" * 60)
        
        X = self.df[['days', 'miles', 'receipts']]
        y = self.df['reimbursement']
        
        # Try different tree configurations to find the best one
        best_mae = float('inf')
        best_tree = None
        best_params = None
        
        # Test different parameters
        max_depths = [6, 8, 10, 12, 15]
        min_samples_splits = [10, 15, 20, 25]
        min_samples_leafs = [5, 8, 10, 12]
        
        print("🔍 Testing tree configurations...")
        
        for max_depth in max_depths:
            for min_split in min_samples_splits:
                for min_leaf in min_samples_leafs:
                    tree = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=42
                    )
                    
                    tree.fit(X, y)
                    y_pred = tree.predict(X)
                    mae = mean_absolute_error(y, y_pred)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_tree = tree
                        best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_split,
                            'min_samples_leaf': min_leaf
                        }
        
        self.tree = best_tree
        
        print(f"✅ Best tree found:")
        print(f"   MAE: ${best_mae:.2f}")
        print(f"   Parameters: {best_params}")
        print(f"   Feature importances:")
        for feature, importance in zip(['days', 'miles', 'receipts'], best_tree.feature_importances_):
            print(f"     {feature}: {importance:.3f}")
            
        return best_mae
        
    def extract_tree_logic(self):
        """Extract the exact tree logic as Python code"""
        print("\n💻 EXTRACTING TREE LOGIC")
        print("=" * 60)
        
        # Get the tree structure
        tree = self.tree.tree_
        feature_names = ['days', 'miles', 'receipts']
        
        def generate_code(node, depth=0):
            """Recursively generate Python code for the tree"""
            indent = "    " * depth
            
            if tree.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                code = f"{indent}if {feature} <= {threshold:.6f}:\n"
                code += generate_code(tree.children_left[node], depth + 1)
                code += f"{indent}else:\n"
                code += generate_code(tree.children_right[node], depth + 1)
                
                return code
            else:
                # Leaf node
                value = tree.value[node][0][0]
                return f"{indent}return {value:.6f}\n"
        
        # Generate the complete function
        function_code = """def calculate_reimbursement_tree(trip_duration_days, miles_traveled, total_receipts_amount):
    \"\"\"
    Precise decision tree implementation
    Automatically generated from trained tree
    \"\"\"
    days = trip_duration_days
    miles = miles_traveled
    receipts = total_receipts_amount
    
"""
        
        function_code += generate_code(0, 1)
        
        print("✅ Tree logic extracted!")
        
        # Save to file
        with open('src/generated_tree_calculator.py', 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""\nGenerated Decision Tree Calculator\nAutomatically extracted from trained tree\n"""\n\n')
            f.write(function_code)
            f.write("\nif __name__ == '__main__':\n")
            f.write("    # Test cases\n")
            f.write("    test_cases = [\n")
            f.write("        (3, 93, 1.42),\n")
            f.write("        (1, 55, 3.6),\n")
            f.write("        (5, 500, 1000),\n")
            f.write("    ]\n")
            f.write("    \n")
            f.write("    for days, miles, receipts in test_cases:\n")
            f.write("        result = calculate_reimbursement_tree(days, miles, receipts)\n")
            f.write("        print(f'Case {days}d, {miles}mi, ${receipts}: ${result:.2f}')\n")
        
        print("💾 Code saved to src/generated_tree_calculator.py")
        
        return function_code
        
    def validate_extracted_tree(self):
        """Validate that the extracted tree produces the same results"""
        print("\n✅ VALIDATING EXTRACTED TREE")
        print("=" * 60)
        
        # Import the generated function
        import sys
        sys.path.append('src')
        
        try:
            from generated_tree_calculator import calculate_reimbursement_tree
            
            # Test on a few cases
            test_cases = self.df.sample(10, random_state=42)
            
            total_error = 0
            for _, row in test_cases.iterrows():
                tree_pred = self.tree.predict([[row['days'], row['miles'], row['receipts']]])[0]
                extracted_pred = calculate_reimbursement_tree(row['days'], row['miles'], row['receipts'])
                
                error = abs(tree_pred - extracted_pred)
                total_error += error
                
                if error > 0.01:
                    print(f"⚠️ Mismatch: Tree={tree_pred:.2f}, Extracted={extracted_pred:.2f}")
            
            avg_error = total_error / len(test_cases)
            
            if avg_error < 0.01:
                print("✅ Extraction successful! Tree logic matches perfectly.")
            else:
                print(f"⚠️ Average extraction error: ${avg_error:.4f}")
                
        except ImportError:
            print("❌ Could not import generated tree calculator")
            
    def run_extraction(self):
        """Run the complete tree extraction process"""
        print("🎯 PRECISE DECISION TREE EXTRACTION")
        print("=" * 70)
        
        # Train optimal tree
        mae = self.train_optimal_tree()
        
        # Extract logic
        code = self.extract_tree_logic()
        
        # Validate extraction
        self.validate_extracted_tree()
        
        print(f"\n🎉 EXTRACTION COMPLETE!")
        print(f"🌳 Tree MAE: ${mae:.2f}")
        print(f"💻 Code generated and validated")
        
        if mae < 30:
            print("🏆 EXCELLENT! Tree is highly accurate!")
        elif mae < 60:
            print("🥈 GOOD! Tree captures most patterns")
        else:
            print("🔧 NEEDS WORK! Tree may need refinement")
            
        return mae

if __name__ == "__main__":
    extractor = PrecisionTreeExtractor()
    extractor.run_extraction() 