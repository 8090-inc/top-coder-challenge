#!/usr/bin/env python3
"""
Repository Cleanup Script
Moves old/experimental files to archive while keeping active components
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Define what to keep (everything else in these directories will be archived)
KEEP_FILES = {
    # Root level - core files
    'calculate_reimbursement.py',
    'requirements.txt',
    'README.md',
    'config.py',
    '.gitignore',
    'run.sh',
    'run.sh.template',
    'eval.sh',
    'generate_results.sh',
    'public_cases.json',
    'private_cases.json',
    'private_results.txt',
    
    # Models - only latest versions
    'models/cluster_models_optimized.py',
    'models/cluster_router.py',
    'models/__init__.py',
    'models/v5_practical_ensemble.py',
    'models/v5_practical_ensemble.pkl',
    'models/train_v5_model.py',
    
    # Data - keep all
    'data/**',
    
    # Docs - keep latest
    'docs/hypothesis.txt',
    
    # Analysis - only active work
    'analysis/09_cents_hash_discovery/**',
    'analysis/10_per_cluster_ensemble/check_baseline_performance.py',  # Useful reference
    
    # Results/reports - keep for reference
    'results/**',
}

# Directories to completely archive
ARCHIVE_DIRS = [
    'analysis/01_initial_exploration',
    'analysis/02_feature_analysis', 
    'analysis/03_receipt_pattern_analysis',
    'analysis/04_model_comparison',
    'analysis/05_clustering_analysis',
    'analysis/06_cluster_optimization',
    'analysis/07_residual_analysis',
    'analysis/08_ensemble_residual_corrector',  # Old ensemble attempts
    'models/v0.5_cluster_based',  # Old model version
    'models/models',  # Duplicate directory
]

# Individual files to archive
ARCHIVE_FILES = [
    'models/cluster_models.py',  # Replaced by cluster_models_optimized.py
    'analysis/10_per_cluster_ensemble/train_per_cluster_ensemble.py',  # v1 not used
    'analysis/10_per_cluster_ensemble/train_per_cluster_ensemble_v2.py',  # v2 not used
    'analysis/10_per_cluster_ensemble/*.pkl',  # Failed model attempts
    'analysis/10_per_cluster_ensemble/*.json',  # Config for failed models
]

def should_keep(file_path):
    """Check if a file should be kept based on KEEP_FILES patterns"""
    file_path_str = str(file_path)
    
    for keep_pattern in KEEP_FILES:
        if keep_pattern.endswith('/**'):
            # Keep entire directory
            dir_pattern = keep_pattern[:-3]
            if file_path_str.startswith(dir_pattern):
                return True
        elif keep_pattern.endswith('**'):
            # Keep directory contents
            dir_pattern = keep_pattern[:-2]
            if dir_pattern in file_path_str:
                return True
        elif keep_pattern == file_path_str:
            # Exact match
            return True
    
    return False

def create_archive_dir():
    """Create timestamped archive directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = Path(f'archive/cleanup_{timestamp}')
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir

def main():
    """Run the cleanup"""
    print("🧹 Repository Cleanup Script")
    print("=" * 60)
    
    archive_dir = create_archive_dir()
    print(f"Archive directory: {archive_dir}")
    print()
    
    moved_count = 0
    
    # 1. Archive entire directories
    print("📁 Archiving old directories...")
    for dir_path in ARCHIVE_DIRS:
        if Path(dir_path).exists():
            dest = archive_dir / dir_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(dir_path, str(dest))
            print(f"  Moved: {dir_path}")
            moved_count += 1
    
    # 2. Archive individual files
    print("\n📄 Archiving individual files...")
    for pattern in ARCHIVE_FILES:
        if '*' in pattern:
            # Handle wildcards
            base_dir = Path(pattern).parent
            file_pattern = Path(pattern).name
            
            if base_dir.exists():
                for file_path in base_dir.glob(file_pattern):
                    if file_path.is_file():
                        dest = archive_dir / file_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(file_path), str(dest))
                        print(f"  Moved: {file_path}")
                        moved_count += 1
        else:
            # Direct file
            if Path(pattern).exists():
                dest = archive_dir / pattern
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(pattern, str(dest))
                print(f"  Moved: {pattern}")
                moved_count += 1
    
    # 3. Clean up empty directories
    print("\n🗑️  Cleaning up empty directories...")
    for root, dirs, files in os.walk('.', topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            if dir_path.exists() and not any(dir_path.iterdir()) and 'archive' not in str(dir_path):
                dir_path.rmdir()
                print(f"  Removed empty: {dir_path}")
    
    print(f"\n✅ Cleanup complete! Moved {moved_count} items to {archive_dir}")
    
    # Show current structure
    print("\n📊 Current repository structure:")
    print("=" * 60)
    
    important_files = [
        "calculate_reimbursement.py - Main entry point",
        "models/v5_practical_ensemble.py - Latest production model",
        "models/cluster_models_optimized.py - V3 cluster models",
        "analysis/09_cents_hash_discovery/ - Active work on cents",
        "docs/hypothesis.txt - Research findings",
        "data/ - Training/test data",
        "results/ - Evaluation results"
    ]
    
    for item in important_files:
        print(f"  ✓ {item}")
    
    print("\n💡 Next steps:")
    print("  1. Complete cents hash discovery in analysis/09_cents_hash_discovery/")
    print("  2. Use V5 model for predictions (MAE: $77.41)")
    print("  3. Consider implementing a cleaner model versioning system")

if __name__ == "__main__":
    main() 