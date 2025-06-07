"""
Configuration file for the Legacy Reimbursement System Analysis
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PUBLIC_CASES_PATH = RAW_DATA_DIR / "public_cases.json"
PRIVATE_CASES_PATH = RAW_DATA_DIR / "private_cases.json"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
MODELS_DIR = RESULTS_DIR / "models"

# Documentation paths
DOCS_DIR = PROJECT_ROOT / "docs"
HYPOTHESIS_PATH = DOCS_DIR / "hypothesis.txt"
INTERVIEWS_PATH = DOCS_DIR / "INTERVIEWS.md"
PRD_PATH = DOCS_DIR / "PRD.md"

# Analysis paths
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
SCRIPTS_DIR = ANALYSIS_DIR / "scripts"
NOTEBOOKS_DIR = ANALYSIS_DIR / "notebooks"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR,
                  DOCS_DIR, ANALYSIS_DIR, SCRIPTS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Analysis constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Hypothesis status indicators
STATUS_CONFIRMED = "🟢"
STATUS_LIKELY = "🟡"
STATUS_TESTING = "🔵"
STATUS_UNTESTED = "⚪"
STATUS_DISPROVEN = "🔴"
STATUS_UNCLEAR = "❓" 