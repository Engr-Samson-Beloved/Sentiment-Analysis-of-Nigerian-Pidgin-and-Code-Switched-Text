import pandas as pd
from .validation import LABELS
def fleiss_kappa(a) -> dict:
    counts=pd.crosstab(a.text_id,a.sentiment_label).reindex(columns=LABELS,fill_value=0)
    if counts.empty or not counts.sum(axis=1).eq(3).all(): raise ValueError("Three ratings per item required")
    observed=float(((counts**2).sum(axis=1)-3).mean()/6)
    expected=float(((counts.sum()/counts.values.sum())**2).sum())
    return {"fleiss_kappa": (observed-expected)/(1-expected) if expected<1 else None,"observed_agreement":observed,"items":len(counts),"undefined_reason":"Single category marginal" if expected==1 else None}
