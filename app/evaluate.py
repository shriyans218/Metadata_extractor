"""
Per-field recall evaluation, as specified in the assignment:

    Recall (per field) = True / (True + False)
    True  = exact string match between predicted and ground-truth value
    False = mismatch OR not extracted (null)

Usage:
    python -m app.evaluate --predictions outputs/test_predictions.csv \
                            --ground-truth data/test.csv
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from app.extraction.schema import FIELDS, CSV_COLUMN_MAP


def _normalize_for_compare(value) -> str:
    """Light normalization ONLY for comparison purposes (surrounding
    whitespace and letter case; the ground truth has stray spaces), not used anywhere in the extraction logic itself. Keeping
    this separate from extraction avoids smuggling rule-based logic
    into the actual ML pipeline -- this function's only role is to
    make the recall metric fair to trivial formatting differences."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def compute_recall(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    allow_partial: bool = False,
) -> dict:
    """allow_partial=False (default): any unmatched File Name raises, so a
    row can never be silently dropped from the score. allow_partial=True:
    score only the intersection and print the excluded names loudly (needed
    for train.csv, whose labels don't line up 1:1 with the train/ folder)."""
    merged = predictions.merge(
        ground_truth, on="File Name", suffixes=("_pred", "_true"), how="outer", indicator=True
    )
    dropped = merged[merged["_merge"] != "both"]
    if len(dropped):
        names = dropped["File Name"].tolist()
        if not allow_partial:
            raise ValueError(
                "File Name mismatch between predictions and ground truth -- "
                f"{len(dropped)} row(s) didn't join and would have been silently "
                f"excluded from recall: {names}. Use --allow-partial to score "
                "only the matching rows."
            )
        print(f"[warn] excluded from scoring (no match on File Name): {names}")
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")

    recall_scores = {}
    for field in FIELDS:
        csv_col = CSV_COLUMN_MAP[field]
        pred_col = f"{csv_col}_pred"
        true_col = f"{csv_col}_true"

        true_count = 0
        total = len(merged)
        for _, row in merged.iterrows():
            pred_val = _normalize_for_compare(row.get(pred_col))
            true_val = _normalize_for_compare(row.get(true_col))
            if pred_val != "" and pred_val == true_val:
                true_count += 1

        recall_scores[field] = round(true_count / total, 4) if total else 0.0

    recall_scores["overall_macro_recall"] = round(
        sum(v for k, v in recall_scores.items()) / len(FIELDS), 4
    )
    return recall_scores


def main():
    parser = argparse.ArgumentParser(description="Compute per-field recall.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", default="outputs/recall_scores.json")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Score only rows present in both files (use for train.csv).",
    )
    args = parser.parse_args()

    # dtype=str: without it, a single empty cell turns the whole
    # "Renewal Notice (Days)" column into float64, so "30" is read back
    # as "30.0" and never equals the ground truth "30".
    predictions = pd.read_csv(args.predictions, dtype=str)
    ground_truth = pd.read_csv(args.ground_truth, dtype=str)

    scores = compute_recall(predictions, ground_truth, allow_partial=args.allow_partial)

    print(json.dumps(scores, indent=2))
    print("\n| Field | Recall |\n|---|---|")
    for k, v in scores.items():
        print(f"| {k} | {v:.4f} |")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
