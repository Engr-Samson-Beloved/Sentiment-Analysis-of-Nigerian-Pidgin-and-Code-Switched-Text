#!/usr/bin/env python3
"""Train and evaluate the calibrated TF-IDF + linear SVM baseline."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naija_sentiment.benchmark import SENTIMENTS, build_svm_pipeline, save_json, save_model, timed_predictions


def evaluate_split(model, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, list[str]]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    predictions, elapsed = timed_predictions(model, frame["text"])
    labels = list(SENTIMENTS)
    report = classification_report(frame["sentiment"], predictions, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "records": len(frame),
        "accuracy": float(accuracy_score(frame["sentiment"], predictions)),
        "macro_f1": float(f1_score(frame["sentiment"], predictions, labels=labels, average="macro", zero_division=0)),
        "inference_seconds": elapsed,
        "milliseconds_per_record": (elapsed / len(frame) * 1000) if len(frame) else None,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(frame["sentiment"], predictions, labels=labels).tolist(),
    }
    rows = pd.DataFrame(report).transpose().reset_index(names="label")
    return metrics, rows, [str(value) for value in predictions]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = ROOT / "data/processed/research_dataset.csv"
    if not default_input.exists():
        default_input = ROOT / "data/processed/naijasenti_combined.csv"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", help="Optional source_language value to train on; defaults to Nigerian Pidgin when present, otherwise all rows")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Processed dataset not found: {args.input}\nRun python scripts/prepare_data.py first.", file=sys.stderr)
        return 1
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    selected_language = args.language
    if selected_language is None and data["source_language"].eq("Nigerian Pidgin").any():
        selected_language = "Nigerian Pidgin"
    selected = data.loc[data["source_language"].eq(selected_language)].copy() if selected_language else data.copy()
    partitions = {name: selected.loc[selected["original_split"].eq(name)].copy() for name in ("train", "validation", "test")}
    if any(part.empty for part in partitions.values()):
        sizes = {name: len(part) for name, part in partitions.items()}
        raise ValueError(f"Selected dataset must contain non-empty train, validation, and test splits: {sizes}")
    model = build_svm_pipeline(random_seed=args.seed, calibrated=True)
    started = time.perf_counter()
    model.fit(partitions["train"]["text"], partitions["train"]["sentiment"])
    training_seconds = time.perf_counter() - started
    validation_metrics, _, _ = evaluate_split(model, partitions["validation"])
    test_metrics, report_frame, test_predictions = evaluate_split(model, partitions["test"])
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts / "svm_sentiment_pipeline.joblib"
    model_size = save_model(model_path, model)
    metrics = {
        "schema_version": 1,
        "model": "word+character TF-IDF with calibrated LinearSVC",
        "primary_language": selected_language or "All supplied languages",
        "source_dataset": ", ".join(sorted(selected["source_dataset"].astype(str).unique())),
        "random_seed": args.seed,
        "number_of_training_records": len(partitions["train"]),
        "training_seconds": training_seconds,
        "model_size_bytes": model_size,
        "model_size_megabytes": model_size / (1024 * 1024),
        "hardware": platform.platform(),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    save_json(args.artifacts / "metrics.json", metrics)
    report_frame.to_csv(args.reports / "classification_report.csv", index=False)
    predictions = partitions["test"][["text_id", "sentiment", "source_language", "original_split"]].copy()
    predictions["prediction"] = test_predictions
    predictions.to_csv(args.reports / "test_predictions.csv", index=False)
    matrix = pd.DataFrame(test_metrics["confusion_matrix"], index=SENTIMENTS, columns=SENTIMENTS)
    plt.figure(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Nigerian Pidgin test confusion matrix")
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")
    plt.tight_layout()
    plt.savefig(args.reports / "confusion_matrix.png", dpi=180)
    plt.close()
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
