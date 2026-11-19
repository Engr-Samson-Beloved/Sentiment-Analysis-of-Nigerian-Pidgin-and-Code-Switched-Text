"""Validation and local import of a team-supplied research dataset."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .benchmark import SENTIMENTS, prepare_frames

TEXT_COLUMNS = ("text", "tweet", "sentence", "content")
LABEL_COLUMNS = ("sentiment", "label", "sentiment_label")
LANGUAGE_COLUMNS = ("source_language", "language", "language_mix")
SPLIT_COLUMNS = ("original_split", "split", "dataset_split")


def _find_column(columns, candidates, required=True):
    lookup = {str(column).strip().casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    if required:
        raise ValueError(f"Missing required column. Expected one of {list(candidates)}; found {list(columns)}")
    return None


def standardize_manual_frame(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Map a supplied CSV to the research schema without inventing labels or splits."""
    if frame.empty:
        raise ValueError("Uploaded dataset is empty")
    if not dataset_name.strip():
        raise ValueError("Dataset name/source is required")
    text_column = _find_column(frame.columns, TEXT_COLUMNS)
    label_column = _find_column(frame.columns, LABEL_COLUMNS)
    split_column = _find_column(frame.columns, SPLIT_COLUMNS)
    language_column = _find_column(frame.columns, LANGUAGE_COLUMNS, required=False)
    sentiments = frame[label_column].fillna("").astype(str).str.strip().str.casefold()
    if sentiments.str.fullmatch(r"\d+").any():
        raise ValueError("Numeric sentiment labels are ambiguous. Convert them to positive, negative, or neutral before upload.")
    invalid = sorted(set(sentiments) - set(SENTIMENTS))
    if invalid:
        raise ValueError(f"Invalid sentiment labels: {invalid}. Allowed labels: {list(SENTIMENTS)}")
    splits = frame[split_column].fillna("").astype(str).str.strip().str.casefold().replace({"dev": "validation", "val": "validation"})
    invalid_splits = sorted(set(splits) - {"train", "validation", "test"})
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}. Use train, validation, or test.")
    texts = frame[text_column].fillna("").astype(str)
    languages = frame[language_column].fillna("Unspecified").astype(str).str.strip() if language_column else pd.Series("Unspecified", index=frame.index)
    identifiers = [
        hashlib.sha256(f"{dataset_name}|{split}|{index}|{text}".encode("utf-8")).hexdigest()[:20]
        for index, (text, split) in enumerate(zip(texts, splits))
    ]
    return pd.DataFrame(
        {
            "text_id": identifiers,
            "text": texts,
            "sentiment": sentiments,
            "source_language": languages.replace("", "Unspecified"),
            "source_dataset": dataset_name.strip(),
            "original_split": splits,
        }
    )


def import_manual_dataset(
    source: str | Path | BinaryIO,
    *,
    dataset_name: str,
    permission_basis: str,
    licence_or_reference: str,
    confirmed: bool,
    output_path: str | Path,
    provenance_path: str | Path,
    report_path: str | Path,
) -> dict:
    """Validate, clean, and save one authorised team-supplied CSV locally."""
    if not confirmed:
        raise ValueError("Confirm the dataset-use disclaimer before importing data")
    if permission_basis not in {"Team-created", "Consent obtained", "Licensed dataset", "Public-domain dataset"}:
        raise ValueError("Select a valid permission basis")
    if not licence_or_reference.strip():
        raise ValueError("Provide a licence, consent record, or source/citation reference")
    raw = pd.read_csv(source, keep_default_na=False)
    standardized = standardize_manual_frame(raw, dataset_name)
    prepared, cleaning_report = prepare_frames([standardized])
    output = Path(output_path); provenance = Path(provenance_path); report = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    report.write_text(json.dumps(cleaning_report, indent=2, ensure_ascii=False), encoding="utf-8")
    record = {
        "dataset_name": dataset_name.strip(),
        "permission_basis": permission_basis,
        "licence_consent_or_source_reference": licence_or_reference.strip(),
        "confirmed_by_operator": True,
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_received": len(raw),
        "rows_retained": len(prepared),
        "stored_path": str(output),
        "disclaimer": "The research team is responsible for permission, privacy review, citation, and compliance with dataset restrictions.",
    }
    provenance.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"provenance": record, "cleaning_report": cleaning_report}
