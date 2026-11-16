"""Descriptive language-group disparity summaries; not causal fairness claims."""


def language_gap(evaluation,min_samples=10):
    """Compare eligible per-language macro-F1 values and expose excluded groups."""
    eligible={name:values["macro_f1"] for name,values in evaluation["per_language"].items() if values["n"]>=min_samples and values["macro_f1"] is not None}
    excluded={name:values["n"] for name,values in evaluation["per_language"].items() if name not in eligible}
    gap=max(eligible.values())-min(eligible.values()) if len(eligible)>=2 else None
    return {"macro_f1_gap":gap,"eligible_groups":eligible,"excluded_small_groups":excluded,"minimum_samples":min_samples,"interpretation":"Descriptive association only; source, class balance, topic, and annotation quality may confound differences."}
