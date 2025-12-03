"""
Session 2 – Deeper EDA.

Reads `train_data.csv` and produces:
- Summary statistics
- Simple correlation matrix
- A few scatter plots to explore relationships
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)


def main() -> None:
    train_path = PROC / "train_data.csv"
    if not train_path.exists():
        raise FileNotFoundError("Run 01_project_setup.py first to create train_data.csv")

    train = pd.read_csv(train_path)
    print("Train shape:", train.shape)
    print(train.head())

    # Summary stats
    summary = train.describe()
    summary.to_csv(RESULTS / "02_summary_stats.csv")
    print("Saved summary stats to results/02_summary_stats.csv")

    # Correlation heatmap
    corr = train.corr(numeric_only=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.tight_layout()
    plt.savefig(RESULTS / "02_corr_heatmap.png", dpi=150)
    plt.close()
    print("Saved correlation heatmap to results/02_corr_heatmap.png")

    # Scatter plots vs target
    for col in ["trip_duration_days", "miles_traveled", "total_receipts_amount"]:
        plt.figure(figsize=(5, 4))
        sns.scatterplot(data=train, x=col, y="reimbursement", alpha=0.6)
        plt.title(f"reimbursement vs {col}")
        plt.tight_layout()
        plt.savefig(RESULTS / f"02_scatter_{col}.png", dpi=150)
        plt.close()

    print("EDA plots saved in results/")


if __name__ == "__main__":
    main()
