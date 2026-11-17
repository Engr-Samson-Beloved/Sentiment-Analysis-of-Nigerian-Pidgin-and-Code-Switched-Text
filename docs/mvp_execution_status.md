# CTE 442 MVP Execution Status

Verified: 2026-11-17

## Current stage

The visible MVP is complete and running locally. The official NaijaSenti benchmark was downloaded from the `HausaNLP/NaijaSenti-Twitter` repository's converted Parquet revision, cleaned, evaluated, and exposed through the five-tab Streamlit application.

## Downloaded benchmark

| Language | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Nigerian Pidgin (`pcm`) | 5,121 | 1,281 | 4,154 | 10,556 |
| Hausa (`hau`) | 14,172 | 2,677 | 5,303 | 22,152 |
| Yoruba (`yor`) | 8,522 | 2,090 | 4,515 | 15,127 |
| **Total** | **27,815** | **6,048** | **13,972** | **47,835** |

The preparation stage retained 45,458 rows after removing 2,377 duplicate language/text pairs. No missing or empty rows were found. The cleaned language totals are Hausa 21,937; Nigerian Pidgin 8,631; Yoruba 14,890.

## Baseline result

The word-and-character TF-IDF calibrated Linear SVM trained on 4,609 deduplicated official Nigerian Pidgin training rows.

| Split | Records | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Validation | 996 | 0.6747 | 0.3601 |
| Test | 3,026 | 0.5744 | 0.3476 |

Training took approximately 1.18 seconds. The model is 4.36 MB. Test inference took approximately 0.29 seconds.

The baseline predicted no neutral test items correctly. The deduplicated training partition contains only 66 neutral examples, compared with 3,003 negative and 1,540 positive examples. This is a major limitation and a priority for error analysis; accuracy alone overstates model quality.

## Verification

- Preprocessing and training were rerun successfully from the downloaded files.
- The test suite passed: **38 tests in 6.66 seconds**.
- Streamlit's application test runner executed all five tabs with no application exceptions.
- The processed dataset, model pipeline, metrics JSON, classification report, test predictions, and 1260×900 confusion-matrix image exist.
- Three live sample predictions executed with calibrated probabilities.
- Pilot storage was tested against a temporary CSV; no synthetic submission was added to the real pilot count.

## Next academic priorities

1. Present the MVP as a benchmark pipeline, not as a successful final classifier.
2. Explain the neutral-class scarcity and zero neutral recall in the presentation.
3. Collect the original consented pilot dataset separately and keep its count distinct from NaijaSenti.
4. Complete three-independent-annotator labeling, Fleiss' kappa, and adjudication for original data.
5. Add class-aware experiments without modifying the official test split, then implement fastText, BiLSTM, mBERT, AfroXLMR, and controlled few-shot LLM comparisons.
6. Add repeated seeds, learning curves, confidence intervals, and analysis of sarcasm, slang, spelling variation, and language mix.
