# Implementation Status Report

Last audited: 2026-11-16. This report distinguishes execution evidence from source-code presence.

## Verified working

- CSV schema, required-value, label, and unique-ID validation.
- Unicode-preserving cleaning and common URL, email, handle, and phone redaction.
- Exact/near-duplicate and explicit related-group clustering.
- Exactly-three-distinct-annotator validation, sentiment Fleiss' kappa, majority consensus, and manual queues for disagreement and three-way ties.
- Permission-ledger validation and blocking audit for missing, pending, or rejected sources.
- Conservative privacy screen with an explicit mandatory-human-review limitation.
- Deterministic group-aware 70/15/15 splitting, conflicting-label rejection, leakage tests, stable hashes, and split manifests.
- Accuracy, fixed-three-class macro F1, per-language reporting threshold, confusion matrix values, experiment-record validation, error tables, and descriptive language-gap summaries.
- Data-only CLI commands: `init-templates`, `validate`, `agreement`, `adjudicate`, `audit-permissions`, `audit-privacy`, and `split`.

Evidence: 25 standard-library unit tests passed in 2.975 seconds using Python 3.12 plus the pre-existing pandas installation. A synthetic CLI run generated templates, validation statistics, agreement output, consensus, permission/privacy audits, and 66/12/12 train/validation/test files with a manifest. These synthetic results are infrastructure checks, not research findings.

## Implemented but not executable in this environment

- TF-IDF plus scikit-learn `LinearSVC`, word n-gram configuration, joblib serialization/reload, validation predictions, shared experiment JSON, and confusion-matrix CSV.
- Pytest suite covering model training and serialization.
- Generic learning-curve driver.

The project `.venv` contains only pip. Package installation repeatedly failed because the package index could not be resolved. The direct `train-svm` attempt failed at missing `joblib`; scikit-learn, PyYAML, and pytest are also unavailable. Therefore no SVM metric or end-to-end model result is verified.

## Explicit placeholders or missing work

- fastText, trained-embedding BiLSTM, multilingual BERT, AfroXLMR, and few-shot LLM adapters raise `NotImplementedError` and have no results.
- No real research dataset has been collected, licensed, de-identified, or annotated by the team.
- No confidence intervals, repeated-seed neural runs, model-specific hyperparameter searches, plotted learning curves, sarcasm/slang/spelling tags, or final disparity analysis has been executed.
- Final-test evaluation remains intentionally locked in the initial CLI.

## Next milestones

1. Restore dependency access or supply a compatible offline wheelhouse; run `python scripts/setup_and_verify.py` until pytest and the SVM smoke pipeline pass.
2. Obtain institutional/ethics approval, finalize source permissions, pilot annotation, revise guidelines, then collect and de-identify at least 8,000 texts. Freeze a versioned dataset only after permission, privacy, completeness, agreement, and leakage audits pass.
3. Establish the validated SVM result and learning curve, then implement fastText and BiLSTM as cheaper non-transformer comparisons.
4. Implement mBERT and AfroXLMR with fixed splits, multiple seeds, recorded hardware/runtime, and validation-only selection.
5. Run an approved few-shot LLM benchmark using training-only demonstrations. Never upload private text without explicit permission and an approved protocol.

## Manual team responsibilities

Humans must secure ethics review and consent/license evidence; recruit and train independent annotators; inspect contextual identifiers; adjudicate unresolved labels; review language categories and duplicate clusters; freeze the test set; approve any external-service data transfer; and interpret sarcasm, dialect, slang, and disparity results. Automated checks cannot perform these duties.
