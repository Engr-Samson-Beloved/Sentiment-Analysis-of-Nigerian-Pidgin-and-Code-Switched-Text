"""Local, consent-gated pilot submission storage."""

from __future__ import annotations

import csv
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

PILOT_COLUMNS = ["submission_id", "submitted_at_utc", "text", "language_mix", "topic", "consent_confirmed", "sentiment_suggestion"]
PROHIBITED_PATTERNS = {
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.\w+"),
    "username/handle": re.compile(r"(?<!\w)@\w+"),
    "phone number": re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d"),
    "matric number": re.compile(r"\b(?:matric|mat(?:ric)?\s*no)[:#\s-]*[a-z0-9/-]{4,}\b", re.IGNORECASE),
}


def validate_pilot_text(text: str) -> None:
    if not text.strip():
        raise ValueError("Text is required")
    findings = [name for name, pattern in PROHIBITED_PATTERNS.items() if pattern.search(text)]
    if findings:
        raise ValueError("Remove personal identifiers before submitting: " + ", ".join(findings))


def save_pilot_submission(path: str | Path, *, text: str, language_mix: str, topic: str, consent_confirmed: bool, sentiment_suggestion: str) -> str:
    """Append one consented, identifier-screened submission to local CSV storage."""
    if not consent_confirmed:
        raise ValueError("Consent confirmation is required")
    validate_pilot_text(text)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission_id = uuid.uuid4().hex
    row = {
        "submission_id": submission_id,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "text": text.strip(),
        "language_mix": language_mix,
        "topic": topic.strip(),
        "consent_confirmed": True,
        "sentiment_suggestion": sentiment_suggestion,
    }
    exists = destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return submission_id
