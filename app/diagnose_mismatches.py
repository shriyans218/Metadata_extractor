"""
Quick diagnostic: print predicted vs. ground-truth values side by side
for every mismatch, so you can see WHY recall dropped (wrong extraction
vs. formatting mismatch) instead of just seeing the score.

Usage:
    python -m app.diagnose_mismatches --predictions outputs/test_predictions.csv \
                                       --ground-truth data/test.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from app.extraction.schema import FIELDS, CSV_COLUMN_MAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions, dtype=str)
    gt = pd.read_csv(args.ground_truth, dtype=str)
    merged = pred.merge(gt, on="File Name", suffixes=("_pred", "_true"))

    for field in FIELDS:
        csv_col = CSV_COLUMN_MAP[field]
        pred_col, true_col = f"{csv_col}_pred", f"{csv_col}_true"
        print(f"\n=== {field} ===")
        for _, row in merged.iterrows():
            p = str(row.get(pred_col)).strip()
            t = str(row.get(true_col)).strip()
            mark = "OK " if p.lower() == t.lower() else "MISS"
            if mark == "MISS":
                print(f"[{mark}] {row['File Name']}")
                print(f"    predicted: {p!r}")
                print(f"    expected:  {t!r}")


if __name__ == "__main__":
    main()
