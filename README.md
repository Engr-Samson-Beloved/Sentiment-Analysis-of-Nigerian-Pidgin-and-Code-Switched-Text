# Group 4 — Nigerian multilingual sentiment research
Yaba College of Technology · 13 members

Research question: Do multilingual transformers outperform classical and recurrent models on Nigerian Pidgin and English–Yoruba / English–Hausa code-switching, and where do they fail?

## Installation
Use Python 3.11+ on macOS, Linux or Windows. From this project directory:
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
Windows PowerShell: use `py -3.11 -m venv .venv`, then `.venv\Scripts\Activate.ps1` instead. No GPU is needed for the SVM.

## Reproduce the infrastructure demonstration
All supplied samples and ratings are artificial; their scores are not research findings.
```sh
python scripts/make_fixture.py
pytest -q
naija-sentiment init-templates --output data/processed/templates
naija-sentiment validate --input data/samples/synthetic.csv --output results/metrics/validation
naija-sentiment audit-permissions --input data/samples/synthetic.csv --permissions data/samples/synthetic_permissions.csv --output results/metrics/permissions
naija-sentiment audit-privacy --input data/samples/synthetic.csv --output results/metrics/privacy
naija-sentiment agreement --input data/samples/synthetic.csv --annotations data/samples/synthetic_annotations.csv --output data/processed/agreement
naija-sentiment adjudicate --input data/samples/synthetic.csv --annotations data/samples/synthetic_annotations.csv --output data/processed/adjudicated
naija-sentiment split --input data/processed/adjudicated/consensus.csv --output data/processed/splits --dataset-config configs/dataset.yaml
naija-sentiment train-svm --train data/processed/splits/train.csv --validation data/processed/splits/validation.csv --config configs/experiments.yaml --output results/metrics/svm
naija-sentiment evaluate --input data/processed/splits/validation.csv --model results/metrics/svm/model.joblib --output results/metrics/reloaded
```
Only load trusted joblib files. Every command requires an explicit output directory; use a new directory per experiment to preserve earlier outputs.

## Architecture and status
`src/naija_sentiment/data` contains CSV validation, conservative cleaning, transitive duplicate grouping, deterministic grouped splitting, annotation checks, agreement, permission enforcement, privacy screening and version manifests. `models/svm.py` implements TF-IDF + LinearSVC; evaluation is model-independent. CLI reports include JSON metrics, a confusion-matrix CSV, CSV predictions, configuration, provenance and the serialized pipeline. Word n-grams and C are configurable.

The five other model modules expose explicit unimplemented interfaces. Generic learning-curve execution, error tables and descriptive language-gap summaries now exist, but they have not been exercised with a trained model in this environment. Confidence intervals, neural training and plotted figures remain next-phase work. Add adapters exposing `predict(texts)` and use the shared experiment schema. No external LLM calls are implemented.

## Research preparation
Read all documents in `docs/` before collecting. Target at least 8,000 permitted, de-identified examples. Source permissions are recorded in a separate restricted ledger. Templates do not certify permission. Every item requires three distinct human annotators; identity checks cannot prove real-world independence. Disagreements enter a queue; three-way ties retain blank labels and must be resolved before splitting. Preserve original ratings and record adjudicator decisions separately.

Use `group_id` for shared conversation, author or other known relationships (opaque local IDs). Near-duplicate grouping is conservative, quadratic and heuristic; manually review clusters and undetected paraphrases. Group counts drive approximate 70/15/15 ratios, so row ratios can differ. Small strata trigger a warning. Inspect all split distributions. Never tune on test data; this initial evaluator accepts validation only.

Cleaning redacts common email, URL, handle and phone patterns while retaining Unicode and punctuation. It is not complete anonymization: trained reviewers must remove names, locations and contextual identifiers before research use. Raw data and outputs are Git-ignored. Keep a restricted, backed-up data store outside Git. Record permission evidence, dataset hashes, split manifests, dependency lock, seed and hardware for each release.

## Next milestones
Pilot annotation and revise guidelines; approve collection permissions; curate 8,000 items; freeze grouped splits; tune baseline only on validation; implement neural adapters and learning curves; preregister final evaluation; write error and disparity analysis. See team workflow for responsibilities.

## One-command setup and verification
After activating `.venv`, run:
```sh
python scripts/setup_and_verify.py
```
This installs dependencies, runs pytest, exercises the complete CLI workflow on the committed synthetic fixtures, verifies reloaded predictions, and records a dependency snapshot. Each execution writes logs and a machine-readable status into a new `results/metrics/smoke-*` directory. A failed installation stops the workflow without claiming model verification.

For offline installation, obtain compatible Python 3.12/macOS arm64 wheels (including build dependencies setuptools and wheel) on an appropriately connected machine, then run:
```sh
python scripts/setup_and_verify.py --wheelhouse /path/to/wheels
```
Use `--skip-install` only when dependencies are already installed.
