"""Learning-curve utilities. Execution requires scikit-learn."""


def training_sizes(total,points=(.1,.25,.5,.75,1.0)):
    """Return unique, increasing nested sample sizes."""
    if total<1: raise ValueError("Training set is empty")
    return sorted({max(1,min(total,round(total*fraction))) for fraction in points})


def learning_curve(train_df,validation_df,trainer,config):
    """Train fresh models on seeded nested subsets; never consumes test data."""
    import random
    from .metrics import metrics
    seed=config.get("seed",42); indices=list(range(len(train_df))); random.Random(seed).shuffle(indices)
    results=[]
    for size in training_sizes(len(train_df)):
        subset=train_df.iloc[indices[:size]]
        if subset.sentiment.nunique()<2: continue
        model=trainer(subset,config); predicted=model.predict(validation_df.text)
        score=metrics(validation_df,predicted)
        results.append({"training_rows":size,"accuracy":score["accuracy"],"macro_f1":score["macro_f1"]})
    return results
