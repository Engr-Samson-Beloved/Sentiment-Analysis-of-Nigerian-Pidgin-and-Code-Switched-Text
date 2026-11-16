"""Shared experiment record used by every model family."""

REQUIRED_FIELDS={"schema_version","model_identifier","dataset_version","split_identifier","training_configuration","random_seed","accuracy","macro_f1","per_language","prediction_file","runtime_seconds","hardware"}


def experiment_record(*,model_identifier,dataset_version,split_identifier,training_configuration,random_seed,evaluation,prediction_file,runtime_seconds,hardware):
    record={"schema_version":1,"model_identifier":model_identifier,"dataset_version":dataset_version,"split_identifier":split_identifier,"training_configuration":training_configuration,"random_seed":random_seed,"accuracy":evaluation["accuracy"],"macro_f1":evaluation["macro_f1"],"per_language":evaluation["per_language"],"prediction_file":str(prediction_file),"runtime_seconds":runtime_seconds,"hardware":hardware,"labels":evaluation["labels"],"class_support":evaluation["class_support"],"confusion_matrix":evaluation["confusion_matrix"]}
    validate_experiment_record(record)
    return record


def validate_experiment_record(record):
    missing=REQUIRED_FIELDS-set(record)
    if missing: raise ValueError(f"Missing experiment fields: {sorted(missing)}")
    for metric in ["accuracy","macro_f1"]:
        if not 0<=record[metric]<=1: raise ValueError(f"{metric} must be between 0 and 1")
