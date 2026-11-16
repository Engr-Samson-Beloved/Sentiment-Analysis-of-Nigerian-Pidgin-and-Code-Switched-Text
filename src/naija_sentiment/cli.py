import argparse
import json
import platform
import time
from pathlib import Path
import pandas as pd
from .data.ingestion import read_dataset
from .data.validation import COLUMNS, statistics
from .data.annotation import ANNOTATION_COLUMNS, validate_annotations, adjudicate
from .data.agreement import fleiss_kappa
from .data.permissions import audit_permissions, read_permissions
from .data.privacy import privacy_audit


def _load_config(path, default):
    if not path:
        return default
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required when a YAML configuration is supplied") from exc
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return loaded

def main():
    parser=argparse.ArgumentParser(description="Group 4 research workflow")
    parser.add_argument("command",choices=["init-templates","validate","agreement","adjudicate","audit-permissions","audit-privacy","split","train-svm","evaluate"])
    for name in ["input","output","annotations","permissions","train","validation","model","config","dataset-config"]: parser.add_argument("--"+name)
    args=parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError, ModuleNotFoundError) as exc: parser.exit(2,f"Error: {exc}\n")
def run(a):
    def required(name):
        value=getattr(a,name)
        if not value: raise ValueError(f"--{name} is required")
        return value
    cfg=_load_config(a.config, {"seed":42,"ngram_range":[1,2],"C":1.0})
    dc=_load_config(a.dataset_config, {})
    def read(path,labeled=False): return read_dataset(path,labeled,dc.get("language_categories"))
    out=Path(required("output")); out.mkdir(parents=True,exist_ok=True)
    if a.command=="init-templates":
        pd.DataFrame(columns=COLUMNS+["group_id"]).to_csv(out/"dataset.csv",index=False)
        pd.DataFrame(columns=ANNOTATION_COLUMNS).to_csv(out/"annotations.csv",index=False)
        pd.DataFrame(columns=["source_reference","permission_status","license_or_consent","evidence_location","restrictions"]).to_csv(out/"permissions.csv",index=False)
        return
    if a.command in ["agreement","adjudicate"]:
        df=read(required("input")); ann=pd.read_csv(required("annotations"),dtype=str,keep_default_na=False)
        validate_annotations(ann,df.text_id,dc.get("language_categories"))
        (out/"agreement.json").write_text(json.dumps(fleiss_kappa(ann),indent=2))
        consensus,queue=adjudicate(ann); queue.to_csv(out/"adjudication_queue.csv",index=False)
        if a.command=="adjudicate":
            df.drop(columns=["sentiment","language_mix"]).merge(consensus,on="text_id").to_csv(out/"consensus.csv",index=False)
        return
    if a.command=="audit-permissions":
        df=read(required("input"))
        report=audit_permissions(df, read_permissions(required("permissions")))
        (out/"permission_audit.json").write_text(json.dumps(report,indent=2), encoding="utf-8")
        if not report["ready_for_research_use"]:
            raise ValueError("Permission audit failed; inspect permission_audit.json")
        return
    if a.command=="audit-privacy":
        df=read(required("input"))
        report=privacy_audit(df)
        (out/"privacy_audit.json").write_text(json.dumps(report,indent=2), encoding="utf-8")
        if not report["automated_screen_passed"]:
            raise ValueError("Privacy screen found potential identifiers; inspect privacy_audit.json")
        return
    if a.command in ["validate","split"]:
        df=read(required("input"),a.command=="split")
        (out/"statistics.json").write_text(json.dumps(statistics(df),indent=2))
        if a.command=="split":
            from .data.splitting import split_dataset
            from .data.versioning import split_manifest, write_manifest
            seed=dc.get("seed",42); threshold=dc.get("near_duplicate_threshold",.9)
            parts=split_dataset(df,seed,threshold,dc.get("language_categories"))
            for name,part in parts.items(): part.to_csv(out/f"{name}.csv",index=False)
            write_manifest(out/"split_manifest.json",split_manifest(df,parts,seed,threshold))
        return
    import joblib
    import yaml
    from .data.deduplication import duplicate_groups
    from .models.svm import train
    from .evaluation.metrics import metrics
    from .evaluation.reporting import experiment_record
    from .data.versioning import dataframe_sha256
    started=time.perf_counter()
    if a.command=="train-svm":
        training=read(required("train"),True); df=read(required("validation"),True)
        if not training.split.eq("train").all() or not df.split.eq("validation").all(): raise ValueError("Expected train and validation partitions; test data forbidden")
        combined=pd.concat([training,df],ignore_index=True)
        if combined.text_id.duplicated().any(): raise ValueError("Overlapping IDs")
        groups=duplicate_groups(combined,dc.get("near_duplicate_threshold",.9))
        if set(groups[:len(training)]) & set(groups[len(training):]): raise ValueError("Related examples cross partitions")
        model=train(training,cfg); joblib.dump(model,out/"model.joblib")
        model=joblib.load(out/"model.joblib")
        dataset_version=dataframe_sha256(combined)
        (out/"model_metadata.json").write_text(json.dumps({"dataset_version":dataset_version,"training_rows":len(training),"validation_rows":len(df),"training_configuration":cfg},indent=2),encoding="utf-8")
    else:
        df=read(required("input"),True)
        if not df.split.eq("validation").all(): raise ValueError("Initial evaluation command accepts validation only; final test remains locked")
        model=joblib.load(required("model"))
        metadata_path=Path(required("model")).with_name("model_metadata.json")
        dataset_version=json.loads(metadata_path.read_text(encoding="utf-8"))["dataset_version"] if metadata_path.exists() else dataframe_sha256(df)
    predicted=model.predict(df.text)
    df.assign(prediction=predicted).to_csv(out/"predictions.csv",index=False)
    evaluation=metrics(df,predicted)
    pd.DataFrame(evaluation["confusion_matrix"],index=evaluation["labels"],columns=evaluation["labels"]).to_csv(out/"confusion_matrix.csv",index_label="actual\\predicted")
    result=experiment_record(model_identifier="tfidf-linear-svm",dataset_version=dataset_version,split_identifier="validation",training_configuration=cfg if a.command=="train-svm" else None,random_seed=cfg.get("seed"),evaluation=evaluation,prediction_file="predictions.csv",runtime_seconds=time.perf_counter()-started,hardware=platform.platform())
    (out/"metrics.json").write_text(json.dumps(result,indent=2))
    (out/"config.yaml").write_text(yaml.safe_dump(cfg))
if __name__=="__main__": main()
