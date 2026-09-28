"""
Batch prediction pipeline: run extraction over every document in a
directory (e.g. data/test/) and write out predictions.csv in the same
column format as the provided train.csv / test.csv, so it can be fed
straight into evaluate.py.

Usage:
    python -m app.predict --input-dir data/test --output outputs/test_predictions.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from app.loaders import load_documents_from_dir
from app.extraction import extract_fields, FIELDS, CSV_COLUMN_MAP

load_dotenv(override=True)


def run_predictions(input_dir: str, self_consistency: bool = True) -> pd.DataFrame:
    documents = load_documents_from_dir(input_dir)

    rows = []
    for file_name, text in documents.items():
        print(f"Extracting: {file_name} ({len(text)} chars)...")
        if not text.strip():
            print(f"  [warn] empty text extracted for {file_name}, skipping LLM call")
            row = {"File Name": file_name}
            for f in FIELDS:
                row[CSV_COLUMN_MAP[f]] = None
            rows.append(row)
            continue

        result = extract_fields(text, self_consistency=self_consistency)
        values = result.to_dict()

        row = {"File Name": file_name}
        for f in FIELDS:
            row[CSV_COLUMN_MAP[f]] = values[f]["value"]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run batch metadata extraction.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--no-self-consistency",
        action="store_true",
        help="Disable the double-call self-consistency check (faster, less robust).",
    )
    args = parser.parse_args()

    df = run_predictions(args.input_dir, self_consistency=not args.no_self_consistency)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
