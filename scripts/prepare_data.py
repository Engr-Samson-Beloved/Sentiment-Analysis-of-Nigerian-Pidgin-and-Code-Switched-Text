#!/usr/bin/env python3
"""Clean downloaded NaijaSenti CSV files and produce a combined benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naija_sentiment.benchmark import LANGUAGE_CONFIGS, SPLITS, prepare_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/naijasenti_combined.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/dataset_summary.json")
    args = parser.parse_args()
    expected = [args.input / f"naijasenti_{config}_{split}.csv" for config in LANGUAGE_CONFIGS for split in SPLITS]
    missing = [path for path in expected if not path.exists()]
    if missing:
        print("Preparation stopped; missing downloaded files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print("Run: python scripts/download_data.py", file=sys.stderr)
        return 1
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in expected]
    combined, report = prepare_frames(frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(combined):,} rows -> {args.output}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
