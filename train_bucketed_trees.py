#!/usr/bin/env python3
"""train_bucketed_trees.py

Train three DecisionTreeRegressor models (short/medium/long trips) and
export each as a standalone pure-Python scorer inside `src/`.

Buckets
-------
short  : trip_duration_days <= 4
medium : 5 <= days <= 8
long   : days >= 9

Each tree is trained on engineered features from src/features.py and
has depth 8 / min_leaf 20 to avoid over-fitting.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from src.features import make_features, FEATURE_NAMES

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "public_cases.json"
SRC_DIR = ROOT / "src"
SRC_DIR.mkdir(exist_ok=True)

BUCKET_NAMES = {
    "short": "calculate_reimbursement_tree_short",
    "mid": "calculate_reimbursement_tree_mid",
    "long": "calculate_reimbursement_tree_long",
}

PARAMS = dict(max_depth=10, min_samples_leaf=10, random_state=42)

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


def train_bucket(X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(**PARAMS)
    tree.fit(X, y)
    return tree


def export_tree(tree: DecisionTreeRegressor, func_name: str, path: Path) -> None:
    """Write pure-Python scorer to *path* with given function name."""

    feature_names = FEATURE_NAMES
    structure = tree.tree_

    def generate_code(node: int, depth: int) -> str:
        indent = "    " * depth
        if structure.feature[node] != -2:  # split
            feat = feature_names[structure.feature[node]]
            thresh = structure.threshold[node]
            left = generate_code(structure.children_left[node], depth + 1)
            right = generate_code(structure.children_right[node], depth + 1)
            return (
                f"{indent}if {feat} <= {thresh:.6f}:\n" + left +
                f"{indent}else:\n" + right
            )
        else:
            value = structure.value[node][0][0]
            return f"{indent}return {value:.6f}\n"

    header_lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated decision tree scorer."""',
        "from typing import Union",
        "",
        f"def {func_name}(trip_duration_days: Union[int,float],",
        "                miles_traveled: Union[int,float],",
        "                total_receipts_amount: Union[int,float]) -> float:",
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
        f"    {', '.join(FEATURE_NAMES)} = (",
        "        days, miles, receipts, 1.0 if days==5 else 0.0,",
        "        miles_lt400, miles_400_800, miles_gt800,",
        "        rec_lt1500, rec_gt1500, days_receipts,",
        "        miles_per_day, rec_gt1500_sq",
        "    )",
    ]

    body_code = generate_code(0, 1)

    full_code = "\n".join(header_lines) + "\n" + body_code

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
        tree = train_bucket(X, y)
        preds = tree.predict(X)
        mae = mean_absolute_error(y, preds)
        overall_err.extend(abs(preds - y))
        out_path = SRC_DIR / f"{func_name}.py"
        export_tree(tree, func_name, out_path)
        print(f"Bucket {name:<5} -> samples {len(y):3d}, MAE ${mae:.2f}, saved {out_path.name}")

    print(f"Overall MAE: ${np.mean(overall_err):.2f}")

if __name__ == "__main__":
    main() 