"""Generate artificial English fixtures only; never research evidence."""
import random
from pathlib import Path
import pandas as pd
rng=random.Random(42)
rows=[]; annotations=[]
for i in range(90):
    label=["positive","negative","neutral"][i%3]
    words=" ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz",k=8)) for _ in range(8))
    rows.append(dict(text_id=f"synthetic-{i}",text=f"{label} {words}",source_type="synthetic",source_reference="fixture-v1",language_mix="english_only",sentiment=label,split=""))
    for j in range(3): annotations.append(dict(text_id=f"synthetic-{i}",annotator_id=f"synthetic-rater-{j}",sentiment_label=label,language_label="english_only",notes="Artificial rating"))
out=Path("data/samples"); out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/"synthetic.csv",index=False)
pd.DataFrame(annotations).to_csv(out/"synthetic_annotations.csv",index=False)
