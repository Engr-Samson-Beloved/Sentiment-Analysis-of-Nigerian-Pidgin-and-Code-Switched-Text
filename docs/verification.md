# Local verification — 2026-11-16

Environment: macOS arm64, zsh, Apple M1 Max 32-core Metal GPU; no CUDA. System Python 3.9.6; local virtual environment created using available runtime Python 3.12. pip available; uv absent. Git repository initialized without changing global identity.

`pip install -e '.[dev]'` failed while fetching setuptools: DNS/network error `[Errno 8] nodename nor servname provided, or not known`. scikit-learn is absent from the available runtime. `.venv/bin/python -m pytest -q` could not start because pytest is not installed. No SVM performance, full smoke success, or passing pytest suite is claimed.

Syntax compilation passed for src, tests and scripts. Using the existing runtime's pandas with PYTHONPATH=src, executed assertions passed for synthetic CSV schema, Unicode cleaning, duplicate grouping (90 independent fixtures), three-annotator completeness, unanimous Fleiss kappa and consensus generation. Generated 90 artificial texts and 270 artificial ratings; these are testing fixtures only.

After network access is available, install requirements and execute the complete README sequence. Record `python -m pip freeze > requirements-lock.txt` after successful installation. Split/model/serialization/metrics tests remain unexecuted. Definition of done remains pending those tests and the end-to-end SVM smoke run.

## Follow-up installation attempt
Retried dependency installation; package-index DNS resolution still failed. Added and executed `scripts/setup_and_verify.py`, which records a failed status and installation logs rather than proceeding with missing dependencies. Its syntax compilation and help command passed. The script supports a local wheelhouse and preserves each run in a separate output directory. Full test and SVM execution remain blocked by unavailable dependencies.

## Executed data regression audit
Added `tests/test_data_unittest.py` and ran it using the available Python 3.12 runtime with pandas and `PYTHONPATH=src`. Result: **15 tests passed in 0.110 seconds**. Coverage: CSV loading and distributions, missing columns/text, invalid sentiment, duplicate IDs, Unicode preservation and common identifier redaction, exact/near duplicates and related groups, annotation completeness and repeated raters, Fleiss kappa (unanimity, complete disagreement and undefined marginal), consensus, majority and three-way ties. Source, test and script syntax compilation also passed.

At that audit stage these were module-level tests, not proof of a working installed CLI in `.venv`; dependency installation still failed because of DNS resolution. Later follow-up results below supersede the earlier splitting and evaluation status. Five other model adapters remain explicit placeholders.

## Full audit follow-up

After removing unnecessary eager ML imports, data-only CLI commands executed successfully. Permission and privacy audits were added and exercised. Splitting was replaced with a deterministic grouped implementation that has no scikit-learn dependency; it produced 66 training, 12 validation, and 12 test synthetic rows, with balanced synthetic sentiment counts and no group leakage. A version manifest recorded stable hashes.

The final expanded dependency-independent suite passed **25 tests in 2.975 seconds**. It covers permission evidence, privacy screening, split reproducibility, related-group containment, conflicting duplicate labels, dataset hashes, evaluation metrics, experiment schema, error summaries, disparity reporting, and learning-curve sizes in addition to earlier coverage. Source syntax compilation passed.

A direct SVM command failed with the readable blocker `No module named 'joblib'`; `.venv/bin/python -m pytest -q` failed with `No module named pytest`. Installation had already failed due to package-index DNS resolution. SVM training, serialization/reload, and the full installed-package smoke test remain unverified.
