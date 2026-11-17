#!/usr/bin/env python3
"""Download the official NaijaSenti benchmark splits from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naija_sentiment.benchmark import DATASET_ID, LANGUAGE_CONFIGS, SPLITS, standardize_hf_split


PARQUET_REVISION = "refs/convert/parquet"


def parquet_files_by_split(repo_files: list[str], config: str) -> dict[str, list[str]]:
    """Select official auto-converted Parquet files for one configuration."""
    selected = {
        split: sorted(
            path
            for path in repo_files
            if path.startswith(f"{config}/{split}/") and path.endswith(".parquet")
        )
        for split in SPLITS
    }
    missing = [split for split, paths in selected.items() if not paths]
    if missing:
        raise ValueError(f"No converted Parquet files found for {config} splits: {missing}")
    return selected


def load_official_dataset(config: str):
    """Load the dataset, falling back from its legacy script to official Parquet."""
    from datasets import load_dataset

    try:
        return load_dataset(DATASET_ID, config), "dataset_repository"
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        from huggingface_hub import HfApi, hf_hub_download

        print("  Legacy dataset script rejected by this datasets version; using official converted Parquet files.", flush=True)
        repo_files = HfApi().list_repo_files(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=PARQUET_REVISION,
        )
        selected = parquet_files_by_split(repo_files, config)
        local_files = {
            split: [
                hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=filename,
                    repo_type="dataset",
                    revision=PARQUET_REVISION,
                )
                for filename in filenames
            ]
            for split, filenames in selected.items()
        }
        return load_dataset("parquet", data_files=local_files), "official_converted_parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=sorted(LANGUAGE_CONFIGS), default=list(LANGUAGE_CONFIGS))
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        manifest = {"source_dataset": DATASET_ID, "languages": {}, "files": []}
        for config in args.languages:
            print(f"Downloading {DATASET_ID} configuration {config}...", flush=True)
            dataset, loading_method = load_official_dataset(config)
            missing_splits = set(SPLITS) - set(dataset.keys())
            if missing_splits:
                raise ValueError(f"{config} is missing official splits: {sorted(missing_splits)}")
            manifest["languages"][config] = {"loading_method": loading_method, "splits": {}}
            for split_name in SPLITS:
                frame = standardize_hf_split(dataset[split_name], config, split_name)
                destination = args.output / f"naijasenti_{config}_{split_name}.csv"
                frame.to_csv(destination, index=False)
                manifest["languages"][config]["splits"][split_name] = len(frame)
                manifest["files"].append(str(destination.relative_to(ROOT)))
                print(f"  {split_name}: {len(frame):,} rows -> {destination}")
        (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (args.output / "download_error.json").unlink(missing_ok=True)
        print("Download complete.")
        return 0
    except Exception as exc:
        error = {"source_dataset": DATASET_ID, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        (args.output / "download_error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(f"Dataset download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("See README.md > Manual dataset recovery for exact recovery instructions.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
