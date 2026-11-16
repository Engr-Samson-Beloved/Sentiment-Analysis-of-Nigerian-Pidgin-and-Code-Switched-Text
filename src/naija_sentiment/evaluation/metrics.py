from collections import Counter

from ..data.validation import LABELS


def _confusion(actual,predicted):
    positions={label:index for index,label in enumerate(LABELS)}
    matrix=[[0 for _ in LABELS] for _ in LABELS]
    for truth,guess in zip(actual,predicted):
        if truth not in positions or guess not in positions:
            raise ValueError(f"Unknown label in evaluation: actual={truth!r}, predicted={guess!r}")
        matrix[positions[truth]][positions[guess]]+=1
    return matrix


def _macro_f1(actual,predicted):
    scores=[]
    for label in LABELS:
        tp=sum(truth==label and guess==label for truth,guess in zip(actual,predicted))
        fp=sum(truth!=label and guess==label for truth,guess in zip(actual,predicted))
        fn=sum(truth==label and guess!=label for truth,guess in zip(actual,predicted))
        denominator=2*tp+fp+fn
        scores.append(2*tp/denominator if denominator else 0.0)
    return sum(scores)/len(LABELS)


def metrics(df, predicted):
    predicted=list(predicted); actual=df.sentiment.tolist()
    if len(actual)!=len(predicted): raise ValueError("Prediction count does not match dataset row count")
    if not actual: raise ValueError("Cannot evaluate an empty dataset")
    result={"accuracy":sum(a==p for a,p in zip(actual,predicted))/len(actual),"macro_f1":_macro_f1(actual,predicted),"labels":LABELS,"confusion_matrix":_confusion(actual,predicted),"class_support":dict(Counter(actual)),"per_language":{}}
    scored=df.assign(predicted=predicted)
    for lang,g in scored.groupby("language_mix"):
        result["per_language"][lang]={"n":len(g),"macro_f1":_macro_f1(g.sentiment.tolist(),g.predicted.tolist()) if len(g)>=10 else None,"caution":"Descriptive only; inspect class support", "class_support":g.sentiment.value_counts().to_dict()}
    return result
