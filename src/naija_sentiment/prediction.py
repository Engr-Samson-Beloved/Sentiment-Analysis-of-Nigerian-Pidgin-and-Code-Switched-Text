"""Safe loading and prediction helpers for the Streamlit application."""

from pathlib import Path
from typing import Any

def load_pipeline(path: str | Path) -> Any:
    """Load a trusted locally produced joblib pipeline."""
    import joblib

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def predict_sentiment(model: Any, text: str) -> dict[str, Any]:
    """Predict sentiment and optional calibrated confidence for one text."""
    normalized = text.strip()
    if not normalized:
        raise ValueError("Enter non-empty text")
    label = str(model.predict([normalized])[0])
    result: dict[str, Any] = {"sentiment": label, "confidence": None, "probabilities": None}
    if hasattr(model, "predict_proba"):
        values = model.predict_proba([normalized])[0]
        classes = [str(value) for value in model.classes_]
        probabilities = {name: float(value) for name, value in zip(classes, values)}
        result["probabilities"] = probabilities
        result["confidence"] = probabilities[label]
    return result
