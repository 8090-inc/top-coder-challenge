#!/usr/bin/env python3
"""train_bucketed_gbdt.py

Train three GradientBoostingRegressor models (short/medium/long trips) and
export each as a standalone pure-Python scorer inside `src/`.

Buckets
-------
short  : trip_duration_days <= 4
medium : 5 <= days <= 8
long   : days >= 9

Each GBDT is trained on engineered features from src/features.py.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List, Union

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.features import make_features, FEATURE_NAMES

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "public_cases.json"
SRC_DIR = ROOT / "src"
SRC_DIR.mkdir(exist_ok=True)

BUCKET_NAMES = {
    "short": "calculate_reimbursement_gbdt_short",
    "mid": "calculate_reimbursement_gbdt_mid",
    "long": "calculate_reimbursement_gbdt_long",
}

GBDT_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42,
)

def load_dataset() -> List[Tuple[float, float, float, float]]:
    records = []
    with DATA_FILE.open() as f:
        raw = json.load(f)
    for row in raw:
        d = float(row["input"]["trip_duration_days"])
        m = float(row["input"]["miles_traveled"])
        r = float(row["input"]["total_receipts_amount"])
        y = float(row["expected_output"])
        records.append((d, m, r, y))
    return records


def bucket_of(days: float) -> str:
    if days <= 4:
        return "short"
    elif days <= 8:
        return "mid"
    else:
        return "long"


def train_bucket(X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(**GBDT_PARAMS)
    model.fit(X, y)
    return model


def export_gbdt(model: GradientBoostingRegressor, func_name: str, path: Path) -> None:
    """Export a GBDT model as a pure-Python scorer."""
    # This is a simplified exporter for scikit-learn GBDT models.
    # It generates a function that sums the outputs of all trees, scaled by learning_rate.
    feature_names = FEATURE_NAMES
    n_estimators = len(model.estimators_)
    learning_rate = model.learning_rate
    init = float(model.init_.constant_[0]) if hasattr(model.init_, 'constant_') else 0.0

    def tree_to_code(tree, tree_idx):
        structure = tree.tree_
        def node_code(node, depth):
            indent = "    " * depth
            if structure.feature[node] != -2:
                feat = feature_names[structure.feature[node]]
                thresh = structure.threshold[node]
                left = node_code(structure.children_left[node], depth + 1)
                right = node_code(structure.children_right[node], depth + 1)
                return (
                    f"{indent}if {feat} <= {thresh:.6f}:\n" + left +
                    f"{indent}else:\n" + right
                )
            else:
                value = structure.value[node][0][0]
                return f"{indent}return {value:.6f}\n"
        code = f"def tree_{tree_idx}({', '.join(feature_names)}):\n"
        code += node_code(0, 1)
        return code

    # Header and feature engineering
    header_lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated GBDT scorer."""',
        f"def {func_name}(trip_duration_days: Union[int,float],",
        "                miles_traveled: Union[int,float],",
        "                total_receipts_amount: Union[int,float]) -> float:",
        "    # --- BEGIN MAIN FUNCTION ---",
        "    # Always initialize result to 0.0 and return round(result, 2) at the end.",
        "    result = 0.0",
        "    days = float(trip_duration_days)",
        "    miles = float(miles_traveled)",
        "    receipts = float(total_receipts_amount)",
        "    # replicate feature engineering inline for speed",
        "    days_receipts = days * receipts",
        "    miles_lt400 = min(miles, 400.0)",
        "    miles_400_800 = max(0.0, min(miles, 800.0) - 400.0)",
        "    miles_gt800 = max(0.0, miles - 800.0)",
        "    rec_lt1500 = min(receipts, 1500.0)",
        "    rec_gt1500 = max(0.0, receipts - 1500.0)",
        "    miles_per_day = miles / days if days else 0.0",
        "    rec_gt1500_sq = rec_gt1500 ** 2",
        "    # positional binding (names generated for clarity only)",
        f"    {', '.join(feature_names)} = (",
        "        days, miles, receipts, 1.0 if days==5 else 0.0,",
        "        miles_lt400, miles_400_800, miles_gt800,",
        "        rec_lt1500, rec_gt1500, days_receipts,",
        "        miles_per_day, rec_gt1500_sq",
        "    )",
    ]
    # Add each tree as a nested function
    tree_codes = []
    for i, est in enumerate(model.estimators_):
        tree_codes.append(tree_to_code(est[0], i))
    # Call each tree and sum (indent for function body)
    call_lines = [
        f"    result += {learning_rate:.8f} * tree_{i}({', '.join(feature_names)})" for i in range(n_estimators)
    ]
    # Add initial prediction and return (also indented)
    footer_lines = [
        f"    result += {init:.8f}",
        "    return round(result, 2)",
    ]
    # Build the main function body
    import_line = "from typing import Union"
    main_body = [header_lines[0], header_lines[1], import_line, header_lines[2]] + ["    " + line for line in header_lines[3:] + call_lines + footer_lines]
    # Compose the full code
    full_code = "\n".join(main_body) + "\n\n" + "\n\n".join(tree_codes) + "\n"
    path.write_text(full_code)


def main() -> None:
    records = load_dataset()
    buckets: Dict[str, List[Tuple[np.ndarray, float]]] = {k: [] for k in BUCKET_NAMES}

    for d, m, r, y in records:
        b = bucket_of(d)
        buckets[b].append((make_features(d, m, r), y))

    overall_err = []

    for name, func_name in BUCKET_NAMES.items():
        data = buckets[name]
        X = np.array([t[0] for t in data])
        y = np.array([t[1] for t in data])
        model = train_bucket(X, y)
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        overall_err.extend(abs(preds - y))
        out_path = SRC_DIR / f"{func_name}.py"
        export_gbdt(model, func_name, out_path)
        print(f"Bucket {name:<5} -> samples {len(y):3d}, MAE ${mae:.2f}, saved {out_path.name}")

    print(f"Overall MAE: ${np.mean(overall_err):.2f}")

if __name__ == "__main__":
    main() 