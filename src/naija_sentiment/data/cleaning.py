import re
import unicodedata
def clean_text(text: str) -> str:
    """Preserve spelling, case, diacritics, emoji and punctuation; redact common identifiers."""
    text=unicodedata.normalize("NFC", text)
    for pattern, replacement in [(r"https?://\S+", "[URL]"), (r"[\w.+-]+@[\w.-]+\.\w+", "[EMAIL]"), (r"(?<!\w)@\w+", "[USER]"), (r"(?<!\w)\+?\d[\d ()-]{8,}\d", "[PHONE]")]:
        text=re.sub(pattern,replacement,text)
    return " ".join(text.split())
