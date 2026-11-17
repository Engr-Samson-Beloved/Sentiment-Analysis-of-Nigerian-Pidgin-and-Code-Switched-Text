"""NaijaSenti benchmark preparation and reproducible baseline helpers."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DATASET_ID = "HausaNLP/NaijaSenti-Twitter"
LANGUAGE_CONFIGS = {"pcm": "Nigerian Pidgin", "hau": "Hausa", "yor": "Yoruba"}
SPLITS = ("train", "validation", "test")
SENTIMENTS = ("negative", "neutral", "positive")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d")
WHITESPACE_RE = re.compile(r"\s+")


def clean_benchmark_text(value: Any) -> str:
    """Normalize identifiers and whitespace without changing linguistic content."""
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = MENTION_RE.sub("@user", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _find_column(columns: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    lookup = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(f"Could not identify {kind} column. Available columns: {sorted(map(str, columns))}")


def standardize_hf_split(split: Any, config: str, split_name: str) -> pd.DataFrame:
    """Inspect and map one Hugging Face split to the project schema."""
    if config not in LANGUAGE_CONFIGS:
        raise ValueError(f"Unsupported language configuration: {config}")
    frame = split.to_pandas() if hasattr(split, "to_pandas") else pd.DataFrame(split)
    text_column = _find_column(frame.columns, ("tweet", "text", "sentence", "content"), "text")
    label_column = _find_column(frame.columns, ("label", "sentiment", "sentiment_label"), "label")
    labels = frame[label_column]
    feature = getattr(split, "features", {}).get(label_column) if hasattr(split, "features") else None
    if feature is not None and hasattr(feature, "int2str"):
        def decode_label(value: Any) -> str:
            if isinstance(value, str) and not value.strip().isdigit():
                return value.strip().casefold()
            return str(feature.int2str(int(value))).strip().casefold()
        labels = labels.map(decode_label)
    else:
        labels = labels.map(lambda value: str(value).strip().casefold())
    unknown = sorted(set(labels.dropna()) - set(SENTIMENTS))
    if unknown:
        raise ValueError(f"Unexpected sentiment labels for {config}/{split_name}: {unknown}")
    texts = frame[text_column].fillna("").astype(str)
    ids = [
        hashlib.sha256(f"{config}|{split_name}|{index}|{text}".encode("utf-8")).hexdigest()[:20]
        for index, text in enumerate(texts)
    ]
    return pd.DataFrame(
        {
            "text_id": ids,
            "text": texts,
            "sentiment": labels,
            "source_language": LANGUAGE_CONFIGS[config],
            "source_dataset": DATASET_ID,
            "original_split": split_name,
        }
    )


def prepare_frames(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and deduplicate downloaded frames while preserving official splits."""
    if not frames:
        raise ValueError("No raw dataset frames were supplied")
    data = pd.concat(frames, ignore_index=True)
    required = {"text_id", "text", "sentiment", "source_language", "source_dataset", "original_split"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    rows_before = len(data)
    missing_values = int(data["text"].isna().sum() + data["sentiment"].isna().sum())
    data["text"] = data["text"].map(clean_benchmark_text)
    empty_mask = data["text"].eq("") | data["sentiment"].fillna("").astype(str).str.strip().eq("")
    empty_count = int(empty_mask.sum())
    data = data.loc[~empty_mask].copy()
    data["sentiment"] = data["sentiment"].astype(str).str.strip().str.casefold()
    unknown = sorted(set(data["sentiment"]) - set(SENTIMENTS))
    if unknown:
        raise ValueError(f"Unexpected sentiment labels after preparation: {unknown}")
    split_order = pd.Categorical(data["original_split"], categories=list(SPLITS), ordered=True)
    data = data.assign(_split_order=split_order).sort_values("_split_order", kind="stable")
    duplicate_mask = data.duplicated(subset=["source_language", "text"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    data = data.loc[~duplicate_mask].drop(columns="_split_order").reset_index(drop=True)
    report = {
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": len(data),
        "duplicates_removed": duplicate_count,
        "missing_value_count": missing_values,
        "empty_rows_removed": empty_count,
        "class_distribution": data["sentiment"].value_counts().sort_index().to_dict(),
        "language_distribution": data["source_language"].value_counts().sort_index().to_dict(),
        "split_distribution": data["original_split"].value_counts().reindex(SPLITS, fill_value=0).to_dict(),
    }
    return data, report


def build_svm_pipeline(random_seed: int = 42, calibrated: bool = True):
    """Create a word+character TF-IDF linear SVM pipeline."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(preprocessor=clean_benchmark_text, ngram_range=(1, 2), min_df=2, max_features=80_000, sublinear_tf=True)),
            ("character", TfidfVectorizer(preprocessor=clean_benchmark_text, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100_000, sublinear_tf=True)),
        ]
    )
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=random_seed)
    classifier = CalibratedClassifierCV(base, cv=3, method="sigmoid") if calibrated else base
    return Pipeline([("features", features), ("classifier", classifier)])


def timed_predictions(model: Any, texts: pd.Series) -> tuple[Any, float]:
    started = time.perf_counter()
    predictions = model.predict(texts)
    return predictions, time.perf_counter() - started


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_model(path: str | Path, model: Any) -> int:
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination.stat().st_size
