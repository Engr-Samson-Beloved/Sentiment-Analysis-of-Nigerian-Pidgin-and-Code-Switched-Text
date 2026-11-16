import pandas as pd
LABELS = ["positive", "negative", "neutral"]
LANGUAGES = ["pidgin", "pidgin_english", "english_yoruba", "english_hausa", "english_only", "other_mixed"]
COLUMNS = ["text_id", "text", "source_type", "source_reference", "language_mix", "sentiment", "split"]
def validate(df: pd.DataFrame, labeled: bool = False, languages=None) -> None:
    """Validate schema; blank labels are allowed only before annotation."""
    missing = set(COLUMNS)-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty: raise ValueError("Dataset is empty")
    for col in ["text_id", "text", "source_type", "source_reference"]:
        if df[col].fillna("").str.strip().eq("").any(): raise ValueError(f"Missing {col}")
    if df.text_id.duplicated().any(): raise ValueError("Duplicate text_id")
    for col, allowed in [("sentiment", LABELS), ("language_mix", languages or LANGUAGES), ("split", ["train", "validation", "test"])]:
        values=df[col].fillna("")
        if not values.isin(allowed+[""]).all(): raise ValueError(f"Invalid {col}")
        if labeled and col != "split" and values.eq("").any(): raise ValueError(f"Missing {col}")
def statistics(df: pd.DataFrame) -> dict:
    return {c: df[c].value_counts(dropna=False).to_dict() for c in ["sentiment", "language_mix"]}
