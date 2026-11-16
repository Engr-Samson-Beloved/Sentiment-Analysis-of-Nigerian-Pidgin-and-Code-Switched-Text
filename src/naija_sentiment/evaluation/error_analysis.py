"""Model-independent prediction error tables for qualitative review."""


def error_table(dataset,predicted):
    """Return misclassified rows while retaining text and research tags."""
    predicted=list(predicted)
    if len(dataset)!=len(predicted): raise ValueError("Prediction count does not match dataset row count")
    scored=dataset.copy(); scored["prediction"]=predicted; scored["is_correct"]=scored.sentiment.eq(scored.prediction)
    return scored.loc[~scored.is_correct].copy()


def error_summary(dataset,predicted):
    """Summarize errors by true label, predicted label, and language mix."""
    errors=error_table(dataset,predicted)
    if errors.empty: return {"errors":0,"error_rate":0.0,"by_language":{},"confusions":{}}
    confusions=(errors.groupby(["sentiment","prediction"]).size().sort_values(ascending=False))
    return {"errors":len(errors),"error_rate":len(errors)/len(dataset),"by_language":errors.language_mix.value_counts().to_dict(),"confusions":{f"{truth}->{guess}":int(count) for (truth,guess),count in confusions.items()}}
