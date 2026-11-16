"""Stable hashes and manifests for reproducible dataset releases."""

import hashlib
import json
from pathlib import Path


def dataframe_sha256(df):
    """Hash a dataframe using stable column and row ordering."""
    columns=sorted(df.columns)
    stable=df[columns].fillna("").astype(str).sort_values(columns,kind="stable")
    return hashlib.sha256(stable.to_csv(index=False,lineterminator="\n").encode("utf-8")).hexdigest()


def split_manifest(source,splits,seed,threshold):
    """Create a serializable record of a grouped split operation."""
    return {"schema_version":1,"source_sha256":dataframe_sha256(source),"seed":seed,"near_duplicate_threshold":threshold,"splits":{
        name:{"rows":len(part),"independent_groups":int(part.duplicate_group.nunique()),"sha256":dataframe_sha256(part),"sentiment_counts":part.sentiment.value_counts().to_dict(),"language_counts":part.language_mix.value_counts().to_dict()}
        for name,part in splits.items()}}


def write_manifest(path,manifest):
    Path(path).write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
