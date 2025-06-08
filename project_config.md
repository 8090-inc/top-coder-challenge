# Project Configuration - Black Box Legacy Reimbursement System

## Tech Stack
- **Primary Language**: Python 3.8+
- **Data Analysis**: pandas, numpy, matplotlib, seaborn, scipy
- **Testing**: pytest, unittest
- **Performance**: Built-in profiling tools
- **Output**: Shell script (run.sh) for final implementation

## Critical Patterns & Conventions
- Pure functional approach for calculation logic
- Test-driven development using historical data
- Statistical analysis for pattern discovery
- Incremental hypothesis testing
- Performance-first implementation
- No external dependencies in final solution

## File Structure

project/
├── analysis/
│ ├── exploratory_analysis.py
│ ├── pattern_discovery.py
│ └── hypothesis_testing.py
├── src/
│ ├── calculator.py
│ ├── validator.py
│ └── utils.py
├── tests/
│ ├── test_calculator.py
│ └── test_patterns.py
├── notebooks/
│ └── data_exploration.ipynb
├── run.sh
└── eval.sh


## Success Metrics
- Exact matches: >95% (±$0.01)
- Performance: <5 seconds per test case
- Code quality: Clean, documented, testable
- Discovery: Document exact legacy formula
