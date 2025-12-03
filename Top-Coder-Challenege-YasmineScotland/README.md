Legacy Reimbursement System – Student Project
============================================

CSCI/DASC 6020 – Machine Learning Team Project

This repository contains my version of the legacy travel reimbursement
reverse‑engineering project. The goal is to predict reimbursement amounts
from three input variables:

- `trip_duration_days`
- `miles_traveled`
- `total_receipts_amount`

The structure and file naming follow the session‑style layout used by my
teammates, but the code is intentionally a bit lighter and easier to follow.

Project Layout
--------------

```text
project/
├── data/
│   ├── raw/
│   │   └── public_cases.json
│   └── processed/
│       ├── train_data.csv
│       ├── test_data.csv
│       ├── train_features.csv
│       └── test_features.csv
├── models/
│   └── saved/
│       └── final_model.joblib
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── final_report_outline.md
├── results/
│   └── *.csv / *.png
├── 01_project_setup.py
├── 02_deep_eda.py
├── 03_feature_engineering.py
├── 04_baseline_models.py
├── 05_advanced_models.py
├── 06_tuning_and_ensembles.py
├── 07_production_pipeline.py
├── 08_generate_final_report.py
├── predict.py
├── run.sh
└── README.md
```

Quick Start
-----------

1. Place `public_cases.json` in `data/raw/`.
2. Create and activate a virtual environment.
3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the scripts in order (you can always re‑run them after you make edits):

   ```bash
   python 01_project_setup.py
   python 02_deep_eda.py
   python 03_feature_engineering.py
   python 04_baseline_models.py
   python 05_advanced_models.py
   python 06_tuning_and_ensembles.py
   python 07_production_pipeline.py
   ```

5. Once `07_production_pipeline.py` has trained and saved `models/saved/final_model.joblib`,
   you can use the command‑line prediction interface:

   ```bash
   ./run.sh 3 250 180.50
   ```

6. To generate a text version of the final report outline:

   ```bash
   python 08_generate_final_report.py
   ```

Notes
-----

- The scripts are written to be readable and fairly modest in size. They are
  not heavily optimized or auto‑generated.
- You can tweak feature engineering, model choices, and hyperparameters in
  `03_feature_engineering.py`, `05_advanced_models.py`, and
  `06_tuning_and_ensembles.py` and then re‑run the pipeline.
- The final `predict.py` and `run.sh` are designed to be compatible with the
  instructor’s `eval.sh` and `generate_results.sh` scripts.
