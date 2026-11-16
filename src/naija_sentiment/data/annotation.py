import pandas as pd
from .validation import LABELS, LANGUAGES
ANNOTATION_COLUMNS=["text_id","annotator_id","sentiment_label","language_label","notes"]
def validate_annotations(a, ids, languages=None):
    if not set(ANNOTATION_COLUMNS)<=set(a): raise ValueError("Missing annotation columns")
    if set(a.text_id)!=set(ids): raise ValueError("Missing or unknown annotated items")
    if a.annotator_id.str.strip().eq("").any() or a.duplicated(["text_id","annotator_id"]).any(): raise ValueError("Annotators must be distinct")
    if not a.groupby("text_id").size().eq(3).all(): raise ValueError("Exactly three annotators required per item")
    for c,labels in [("sentiment_label",LABELS),("language_label",languages or LANGUAGES)]:
        if not a[c].isin(labels).all(): raise ValueError(f"Incomplete or invalid {c}")
def adjudicate(a):
    rows=[]; queue=[]
    for tid,g in a.groupby("text_id"):
        row={"text_id":tid}
        for col,out in [("sentiment_label","sentiment"),("language_label","language_mix")]:
            counts=g[col].value_counts(); row[out]=counts.index[0] if counts.iloc[0]>=2 else ""
        rows.append(row)
        if g.sentiment_label.nunique()>1 or g.language_label.nunique()>1: queue.extend(g.to_dict("records"))
    return pd.DataFrame(rows),pd.DataFrame(queue,columns=ANNOTATION_COLUMNS)
