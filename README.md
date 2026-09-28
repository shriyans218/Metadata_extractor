# Rental Agreement Metadata Extraction

An AI/ML system that extracts structured metadata from rental agreements
**regardless of their template** — accepting either scanned images
(`.png`) or Word documents (`.docx`) as input.

Extracted fields:

| Field | Description |
|---|---|
| Agreement Value | Monthly rent amount |
| Agreement Start Date | Lease start date |
| Agreement End Date | Lease end date |
| Renewal Notice (Days) | Notice period required before renewal/termination |
| Party One | Lessor / owner |
| Party Two | Lessee / tenant |

## Approach

**No regex, no rule-based/static extraction is used anywhere in this
pipeline**, per the assignment constraint. The system is a generative,
LLM-based extractor with a self-consistency verification layer on top.

### 1. Text acquisition (template-agnostic by design)

- `.docx` files → parsed with `python-docx` (paragraphs *and* table
  cells, since some agreements place dates/parties in tables).
- `.png` / image files → OCR'd with Tesseract, grayscale-preprocessed.

Both paths converge into plain text before extraction, so the model
never sees or depends on the original layout.

### 2. Extraction (hosted LLM, any OpenAI-compatible provider)

The full document text is sent to a hosted instruction model
(provider and model set in `.env`; default `nvidia` / `moonshotai/kimi-k2.6`,
see [Providers](#providers)) with a single structured prompt describing each field **semantically**
(e.g. *"the monetary rent amount agreed upon"*) rather than by keyword
or position. This is what lets one implementation generalize across
completely different contract wordings and layouts without any
per-template logic. The model is instructed to return strict JSON and
to return `null` rather than hallucinate a value it can't support from
the text.

All the sample documents in `data/` are short enough (1K–20K
characters) to fit whole into the model's context window, so no
chunking or retrieval step is needed — adding one would be unnecessary
complexity for documents this size.

### 3. Self-consistency check

A single LLM call can hallucinate a value once. To guard against that:

1. The same document is extracted **twice**, at different sampling
   temperatures (0.0 and 0.4).
2. Any field where both runs agree exactly is kept.
3. Any field where they disagree triggers a **third, tie-breaking
   call**; if it matches either prior answer that value is kept,
   otherwise the first (temperature-0) answer is kept.

This costs 2–3 LLM calls per document instead of 1; disable it with
`--no-self-consistency` for a faster single-call run.

### Why this design over alternatives

- **Over pure regex/rules:** explicitly disallowed by the assignment,
  and wouldn't generalize past the ~10 templates seen here anyway.
- **Over training a custom NER model:** only 10 labeled training
  documents exist — nowhere near enough to fine-tune a transformer
  without severe overfitting. A few-shot/zero-shot LLM avoids that
  entirely.
- **Over embeddings + retrieval:** the documents are short enough to
  fit whole into context, so chunking/retrieval would add complexity
  without improving accuracy. (This is a reasonable next step if
  documents were book-length, discussed under Limitations below.)

## Project structure

```
.
├── app/
│   ├── loaders/          # docx / OCR text acquisition
│   ├── extraction/       # schema + LLM extraction + self-consistency
│   ├── api/              # FastAPI REST wrapper
│   ├── predict.py        # batch prediction CLI
│   └── evaluate.py       # per-field recall CLI
├── data/                 # provided train/ and test/ data (unchanged)
├── notebooks/
│   └── demo.ipynb        # walkthrough notebook
├── outputs/              # test_predictions.csv + recall_scores.json
├── run_all.sh            # predict + evaluate on the test set
├── requirements.txt
└── .env.example
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need Tesseract OCR installed at the system level (for the
`.png` inputs):

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

Copy `.env.example` to `.env`, choose `LLM_PROVIDER` and add that
provider's key, then verify with `python -m app.check_llm`:

```bash
cp .env.example .env
# edit .env: LLM_PROVIDER=nvidia and NVIDIA_API_KEY=...
python -m app.check_llm
```

### Providers

| `LLM_PROVIDER` | Key variable | Default model |
|---|---|---|
| `nvidia` (default) | `NVIDIA_API_KEY` | `moonshotai/kimi-k2.6` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |

Set `LLM_MODEL` to override the model. All providers are called through
the same OpenAI-compatible client.

## Running predictions

Everything for the test set in one step (macOS/Linux/Git Bash):

```bash
./run_all.sh
```

or run the two commands yourself (works on Windows PowerShell too):

```bash
python -m app.predict --input-dir data/test --output outputs/test_predictions.csv
python -m app.evaluate --predictions outputs/test_predictions.csv --ground-truth data/test.csv
```

`predict` writes a CSV with the same columns as `test.csv`; `evaluate` is
explained below.

## Evaluating recall

Per the assignment's metric — `Recall = True / (True + False)` per
field, where `True` is a match against ground truth and `False` is a
mismatch or a null (not extracted). `app.evaluate` (command above) prints
a table and saves `outputs/recall_scores.json`.

Comparison is exact string match after trimming surrounding whitespace
and lower-casing (the ground-truth CSVs contain stray spaces such as
`"Hanumaiah , Vishal Bhardwaj "`). Nothing else is normalised.
`evaluate.py` reads CSVs with `dtype=str`; otherwise one empty cell turns
the renewal column into floats and `"30"` becomes `"30.0"`.

To inspect misses: `python -m app.diagnose_mismatches --predictions ... --ground-truth ...`.

### Results (test set, 4 documents)

Predictions: `outputs/test_predictions.csv`. Scores: `outputs/recall_scores.json`.

| Field | Recall |
|---|---|
| agreement_value | 1.0000 |
| agreement_start_date | 1.0000 |
| agreement_end_date | 0.7500 |
| renewal_notice_days | 1.0000 |
| party_one | 1.0000 |
| party_two | 0.2500 |
| **overall (macro average)** | **0.8333** |

Scores depend on the live model, so they can change with provider and
model; re-running `app.predict` and `app.evaluate` regenerates them.

Known data quirks that cap exact-match recall regardless of the model:

- **`95980236` end date:** the agreement gives no end date, only "11
  months commencing from 1st April 2010". The prompt computes an end date
  from start + duration (28.02.2011); the ground truth says 31.03.2011
  (12 months), so this row misses on exact match. Labelling
  inconsistency, deliberately not special-cased.
- **`95980236` rent:** the agreement contradicts itself ("Rs. 9,000/-
  (Nine thousand and two hundred rupees only)"). The ground truth follows
  the numerals, so the prompt tells the model to prefer the figure over
  the words. This is a judgement call, not a legal rule.
- **Party names:** ground truth contains punctuation artefacts
  (`.B.Kishore`, `V.V.Ravi Kian`) that exact match penalises.
- **Prompt is template-neutral:** no field description names a
  clause from a specific test document.

### Evaluating on the training set

`train.csv` does not line up 1:1 with `train/`: it has a row for
`24158401-Rental-Agreement` (a file that lives in `test/`) and no label for
`46239065-Standard-Rental-Agreement-Rental-With-Performance-Fee`. Strict
mode therefore errors by design; score the overlapping rows with:

```bash
python -m app.predict --input-dir data/train --output outputs/train_predictions.csv
python -m app.evaluate --predictions outputs/train_predictions.csv --ground-truth data/train.csv --output outputs/train_recall.json --allow-partial
```

The excluded names are printed. One train row has an empty renewal notice,
which counts as a miss under the assignment's definition.

## Running the REST API (optional bonus)

```bash
uvicorn app.api.main:app --reload --port 8000
```

Then either visit the interactive docs at `http://localhost:8000/docs`,
or call it directly:

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@data/test/24158401-Rental-Agreement.png"
```

Response:

```json
{
  "file_name": "24158401-Rental-Agreement.png",
  "fields": {
    "agreement_value": {"value": "12000"},
    "agreement_start_date": {"value": "01.04.2008"},
    "agreement_end_date": {"value": "31.03.2009"},
    "renewal_notice_days": {"value": "60"},
    "party_one": {"value": "Hanumaiah"},
    "party_two": {"value": "Vishal Bhardwaj"}
  }
}
```

## Limitations & honest notes

- **OCR quality bounds accuracy on scans.** If Tesseract garbles a
  scanned image, no downstream model can recover information that was
  never legible in the text it received. A production version would
  add OCR confidence scoring and possibly a vision-capable LLM fallback
  for low-quality scans.
- **A few source documents contain literally invalid dates**
  (e.g. `31.11.2009`, `31.02.2011` — November has 30 days, February
  never has 31). The model is asked to transcribe dates as written
  rather than "correct" them, since correcting them would be a
  judgment call the ground truth itself doesn't make consistently.
- **Party name formatting varies** (full legal name with suffixes vs.
  short name) which can cause exact-match recall to understate
  correctness even when the model identifies the right person. A
  fuzzy-match or normalized-name comparison metric would likely be a
  fairer metric in a follow-up iteration, but the assignment's spec
  calls for exact match, so that's what `evaluate.py` implements.
- **Self-consistency costs 2–3 calls per document.** A larger system
  could skip the second run for documents where the first is unambiguous,
  or extract only the disputed fields in the tie-break call.
