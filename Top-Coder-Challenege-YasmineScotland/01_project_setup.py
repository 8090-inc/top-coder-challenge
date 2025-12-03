"""
Session 1 – Project setup and initial data split.

- Flattens the JSON structure
- Renames columns to friendly names
- Creates a 750 / 250 train–test split
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

PROC.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------
    # Load and flatten JSON
    # ------------------------------------------------------------------
    public_path = RAW / "public_cases.json"
    if not public_path.exists():
        raise FileNotFoundError(f"Expected file not found: {public_path}")

    with public_path.open() as f:
        data = json.load(f)

    # Flatten nested keys like input.trip_duration_days
    df = pd.json_normalize(data, sep=".")
    df.columns = [c.split(".")[-1] for c in df.columns]

    # Identify input and target columns
    feature_cols = ["trip_duration_days", "miles_traveled", "total_receipts_amount"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected input columns: {missing}")

    # Anything that looks like the reimbursement output
    if "expected_output" in df.columns:
        target_col = "expected_output"
    else:
        # fall back: choose the last numeric column
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if not numeric_cols:
            raise ValueError("Could not find numeric target column.")
        target_col = numeric_cols[-1]

    df = df[feature_cols + [target_col]].copy()
    df.rename(columns={target_col: "reimbursement"}, inplace=True)

    # Basic cleaning
    df = df.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Train / test split (750 / 250)
    # ------------------------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        train_size=750,
        test_size=250,
        random_state=42,
        shuffle=True,
    )

    train_df.to_csv(PROC / "train_data.csv", index=False)
    test_df.to_csv(PROC / "test_data.csv", index=False)

    print(f"Saved train_data.csv with {len(train_df)} rows")
    print(f"Saved test_data.csv with  {len(test_df)} rows")

    # ------------------------------------------------------------------
    # Quick sanity‑check histograms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    cols = ["trip_duration_days", "miles_traveled", "total_receipts_amount", "reimbursement"]
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(train_df[col], bins=30)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(RESULTS / "01_initial_histograms.png", dpi=150)
    plt.close(fig)
    print("Saved initial histograms to results/01_initial_histograms.png")


if __name__ == "__main__":
    main()
