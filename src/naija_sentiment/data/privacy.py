"""Conservative automated screening before mandatory human privacy review."""

import re

import pandas as pd

PATTERNS = {
    "url": re.compile(r"https?://\S+", re.IGNORECASE),
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.\w+"),
    "user_handle": re.compile(r"(?<!\w)@\w+"),
    "phone": re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d"),
}


def privacy_audit(dataset: pd.DataFrame) -> dict:
    """Find common residual identifiers; this cannot certify anonymization."""
    findings = []
    counts = {name: 0 for name in PATTERNS}
    for text_id, text in dataset[["text_id", "text"]].itertuples(index=False):
        kinds = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
        if kinds:
            findings.append({"text_id": text_id, "finding_types": kinds})
            for kind in kinds:
                counts[kind] += 1
    return {
        "automated_screen_passed": not findings,
        "human_review_still_required": True,
        "finding_counts": counts,
        "flagged_items": findings,
        "limitations": (
            "Pattern matching does not detect names, locations, contextual identifiers, "
            "or every phone/handle format. A trained human must review every item."
        ),
    }
