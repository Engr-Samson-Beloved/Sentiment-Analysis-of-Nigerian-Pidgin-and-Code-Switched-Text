# Sentiment Analysis of Nigerian Pidgin and Code-Switched Text

CTE 442 research MVP · Group 4 · Yaba College of Technology

This repository provides a reproducible dataset pipeline and Streamlit dashboard for investigating whether multilingual models handle Nigerian Pidgin and English–Yoruba or English–Hausa code-switching better than classical and recurrent models, and where they fail.

The dashboard accepts a CSV supplied manually by the research team. Import is blocked until the operator confirms the team created the data, obtained consent, or has a valid licence/public-domain basis and provides a supporting reference. The optional NaijaSenti downloader remains available as one reproducible dataset source.

## What the MVP does

- Downloads official Nigerian Pidgin (`pcm`), Hausa (`hau`) and Yoruba (`yor`) train, validation and test splits.
- Inspects the downloaded schema and maps the actual text and label fields safely.
- Conservatively normalizes URLs, mentions, email addresses, phone numbers and whitespace while preserving vocabulary, emoji and punctuation.
- Preserves every official split and removes duplicate text without moving test examples into training.
- Trains a calibrated word-and-character TF-IDF Linear SVM on the official Nigerian Pidgin training split.
- Reports validation and test accuracy, macro F1, per-class precision/recall, confusion matrix, runtime and model size.
- Provides a five-tab Streamlit dashboard for research context, data exploration, results, live predictions and permission-gated dataset setup.

The existing annotation, Fleiss’ kappa, permission, privacy, grouped-splitting and experiment-provenance modules remain under `src/naija_sentiment/`.

For annotated research data, the installed CLI also provides `init-templates`, `validate`, `agreement`, `adjudicate`, `audit-permissions`, `audit-privacy`, and `split`. See `docs/annotation_guidelines.md`, `docs/data_collection_guide.md`, and `docs/research_protocol.md` before using any dataset.

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

To use a team-supplied CSV, open **Dataset setup** in the dashboard. Required columns are text/tweet, sentiment/label, and split. Sentiment must be `positive`, `negative`, or `neutral`; split must be `train`, `validation`, or `test`. After import, retrain with:

```sh
python scripts/train_baseline.py --input data/processed/research_dataset.csv
```

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
| `data/processed/research_dataset.csv` | Clean active team-supplied dataset |
| `data/manual/dataset_provenance.json` | Local permission/licence confirmation record |
| `artifacts/svm_sentiment_pipeline.joblib` | Complete feature-and-classifier pipeline |
| `artifacts/metrics.json` | Validation/test metrics, timing and provenance |
| `reports/dataset_summary.json` | Cleaning and distribution report |
| `reports/classification_report.csv` | Per-class test precision, recall and F1 |
| `reports/confusion_matrix.png` | Official Nigerian Pidgin test confusion matrix |
| `reports/test_predictions.csv` | Test labels and predictions without text |

Raw, processed, manually supplied data and model-output directories are ignored by Git except for `.gitkeep` files. Do not commit restricted research data.

## Dashboard pages

1. **Project overview** — question, dataset disclaimer, record count, languages and classes.
2. **Dataset explorer** — filters, text search, anonymized examples and distribution charts.
3. **Model results** — actual saved metrics, classification report and confusion matrix.
4. **Live analyser** — calibrated sentiment predictions and documented limitations.
5. **Dataset setup** — local CSV import with mandatory permission/licence confirmation and provenance record.

## Citation and licence

Use of NaijaSenti requires citation of:

> Shamsuddeen Hassan Muhammad et al. (2022). *NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis*. LREC 2022.

Paper: https://aclanthology.org/2022.lrec-1.63/

The current Hugging Face dataset card declares **CC BY-NC-SA 4.0**. Confirm the applicable licence and platform terms for the exact artifact before redistribution or publication. Attribute the authors, respect non-commercial/share-alike restrictions, and do not imply that public tweets are Group 4’s original collection.

## Ethics and limitations

- The team must have a documented legal/ethical basis for every supplied dataset and comply with its licence, consent, privacy, attribution, and redistribution restrictions.
- Automated cleaning cannot detect every contextual identifier, so human privacy review remains mandatory.
- The baseline may fail on sarcasm, slang, spelling variation, mixed sentiment, domain shift and unfamiliar code-switching.
- Confidence is a calibrated model score, not certainty or proof of correctness.
- Nigerian Pidgin is the first implemented experiment. Hausa and Yoruba are downloaded for later multilingual and code-switching comparisons.
- Do not upload restricted research text to hosted LLMs without explicit permission and institutional approval.
- Keep final test results sealed from model tuning and report class/language support alongside metrics.

## Remaining research work

The complete assignment still requires verified dataset provenance, three independent annotations where labels are created by the team, Fleiss’ kappa and adjudication, language-mix labels, repeated-seed experiments, learning curves, fastText, trained-embedding BiLSTM, mBERT, AfroXLMR, controlled few-shot LLM benchmarking, and qualitative error/disparity analysis.
