from difflib import SequenceMatcher
from .cleaning import clean_text
def duplicate_groups(df, threshold: float = .9):
    """Conservative transitive groups; quadratic comparison suited to initial 8k corpus."""
    if not 0 < threshold <= 1: raise ValueError("Threshold must be in (0,1]")
    parent=list(range(len(df)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    texts=[clean_text(t).casefold() for t in df.text]
    for i in range(len(texts)):
        for j in range(i):
            related=("group_id" in df and df.iloc[i].group_id and df.iloc[i].group_id==df.iloc[j].group_id)
            if related or SequenceMatcher(None,texts[i],texts[j]).ratio()>=threshold: parent[root(i)]=root(j)
    return [root(i) for i in range(len(texts))]
