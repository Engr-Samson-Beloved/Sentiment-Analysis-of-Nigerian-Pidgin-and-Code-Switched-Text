# Sentiment Analysis of Nigerian Pidgin and Code-Switched Text

CTE 442 research MVP · Group 4 · Yaba College of Technology

This repository provides a reproducible benchmark pipeline and Streamlit dashboard for investigating whether multilingual models handle Nigerian Pidgin and English–Yoruba or English–Hausa code-switching better than classical and recurrent models, and where they fail.

The MVP uses [HausaNLP/NaijaSenti-Twitter](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter) as a **secondary public benchmark**. NaijaSenti was not collected by Group 4. The group’s original volunteer-contributed dataset is collected separately through a consent-gated local pilot form and must never be combined misleadingly with public benchmark counts.

## What the MVP does

- Downloads official Nigerian Pidgin (`pcm`), Hausa (`hau`) and Yoruba (`yor`) train, validation and test splits.
- Inspects the downloaded schema and maps the actual text and label fields safely.
- Conservatively normalizes URLs, mentions, email addresses, phone numbers and whitespace while preserving vocabulary, emoji and punctuation.
- Preserves every official split and removes duplicate text without moving test examples into training.
- Trains a calibrated word-and-character TF-IDF Linear SVM on the official Nigerian Pidgin training split.
- Reports validation and test accuracy, macro F1, per-class precision/recall, confusion matrix, runtime and model size.
- Provides a five-tab Streamlit dashboard for research context, data exploration, results, live predictions and original pilot collection.

The existing annotation, Fleiss’ kappa, permission, privacy, grouped-splitting and experiment-provenance modules remain under `src/naija_sentiment/`.

For the original-data research workflow, the installed CLI also provides `init-templates`, `validate`, `agreement`, `adjudicate`, `audit-permissions`, `audit-privacy`, and `split`. See `docs/annotation_guidelines.md`, `docs/data_collection_guide.md`, and `docs/research_protocol.md` before collecting volunteer data.

## Installation

Python 3.11 or newer is required.

macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the complete MVP

```sh
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_baseline.py
pytest -q
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Dataset download

The downloader calls:

```python
load_dataset("HausaNLP/NaijaSenti-Twitter", "pcm")
```

and repeats this for `hau` and `yor`. It requires all three official splits. Raw standardized CSV files are written under `data/raw/`; the combined clean file is written to `data/processed/naijasenti_combined.csv`.

Recent `datasets` versions no longer execute the repository’s legacy `NaijaSenti-Twitter.py` script. The downloader detects that specific error and falls back to the repository’s official `refs/convert/parquet` revision through `huggingface_hub`; it never switches to a different dataset.

Expected dataset-card counts for version 1.0.0 are:

| Configuration | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `pcm` | 5,121 | 1,281 | 4,154 | 10,556 |
| `hau` | 14,172 | 2,677 | 5,303 | 22,152 |
| `yor` | 8,522 | 2,090 | 4,515 | 15,127 |

These are reference values from the dataset card, not locally claimed results. The download manifest records the actual observed counts.

### Manual dataset recovery

If `python scripts/download_data.py` fails:

1. Read `data/raw/download_error.json` for the exact exception.
2. Confirm that the machine can access `https://huggingface.co`.
3. Install or upgrade the datasets library with `python -m pip install --upgrade datasets`.
4. On a connected machine, run the same downloader, then copy all `naijasenti_<language>_<split>.csv` files and `download_manifest.json` into `data/raw/`.
5. Do not create substitute rows or change the official split labels.

## Generated outputs

| File | Purpose |
|---|---|
| `data/raw/naijasenti_<language>_<split>.csv` | Standardized official split |
| `data/raw/download_manifest.json` | Actual downloaded row counts and files |
| `data/processed/naijasenti_combined.csv` | Clean combined benchmark with original splits |
| `data/pilot/original_pilot_submissions.csv` | Separate consented local pilot submissions |
| `artifacts/svm_sentiment_pipeline.joblib` | Complete feature-and-classifier pipeline |
| `artifacts/metrics.json` | Validation/test metrics, timing and provenance |
| `reports/dataset_summary.json` | Cleaning and distribution report |
| `reports/classification_report.csv` | Per-class test precision, recall and F1 |
| `reports/confusion_matrix.png` | Official Nigerian Pidgin test confusion matrix |
| `reports/test_predictions.csv` | Test labels and predictions without text |

Raw, processed, pilot and model-output directories are ignored by Git except for `.gitkeep` files. Do not commit private pilot data.

## Dashboard pages

1. **Project overview** — question, disclosure, languages and separate benchmark/original counts.
2. **Dataset explorer** — filters, text search, anonymized examples and distribution charts.
3. **Model results** — actual saved metrics, classification report and confusion matrix.
4. **Live analyser** — calibrated sentiment predictions and documented limitations.
5. **Original pilot data** — local consent-gated submission form that rejects common identifiers.

## Citation and licence

Use of NaijaSenti requires citation of:

> Shamsuddeen Hassan Muhammad et al. (2022). *NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis*. LREC 2022.

Paper: https://aclanthology.org/2022.lrec-1.63/

The current Hugging Face dataset card declares **CC BY-NC-SA 4.0**. Confirm the applicable licence and platform terms for the exact artifact before redistribution or publication. Attribute the authors, respect non-commercial/share-alike restrictions, and do not imply that public tweets are Group 4’s original collection.

## Ethics and limitations

- NaijaSenti is a public secondary benchmark and cannot replace the required original volunteer dataset.
- The pilot form collects no names, usernames, email addresses, phone numbers or matric numbers; automated screening cannot detect every contextual identifier, so human privacy review remains mandatory.
- The baseline may fail on sarcasm, slang, spelling variation, mixed sentiment, domain shift and unfamiliar code-switching.
- Confidence is a calibrated model score, not certainty or proof of correctness.
- Nigerian Pidgin is the first implemented experiment. Hausa and Yoruba are downloaded for later multilingual and code-switching comparisons.
- Do not upload private pilot text to hosted LLMs without explicit participant permission and institutional approval.
- Keep final test results sealed from model tuning and report class/language support alongside metrics.

## Remaining research work

The complete assignment still requires at least 8,000 permitted original texts, three independent annotations per item, Fleiss’ kappa and adjudication, language-mix labels, repeated-seed experiments, learning curves, fastText, trained-embedding BiLSTM, mBERT, AfroXLMR, controlled few-shot LLM benchmarking, and qualitative error/disparity analysis.
