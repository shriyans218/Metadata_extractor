#!/usr/bin/env bash
# Reproduce test predictions + per-field recall in one go.
set -euo pipefail
python -m app.predict --input-dir data/test --output outputs/test_predictions.csv
python -m app.evaluate --predictions outputs/test_predictions.csv --ground-truth data/test.csv
