import pandas as pd
import pytest
import joblib
from naija_sentiment.data.ingestion import read_dataset
from naija_sentiment.data.validation import validate
from naija_sentiment.data.cleaning import clean_text
from naija_sentiment.data.deduplication import duplicate_groups
from naija_sentiment.data.splitting import split_dataset
from naija_sentiment.data.annotation import validate_annotations, adjudicate
from naija_sentiment.data.agreement import fleiss_kappa
from naija_sentiment.models.svm import train
from naija_sentiment.evaluation.metrics import metrics
@pytest.fixture
def df(): return read_dataset("data/samples/synthetic.csv",True)
def test_schema(df):
    validate(df,True)
    with pytest.raises(ValueError): validate(df.drop(columns="text"))
    df.loc[0,"sentiment"]="bad"
    with pytest.raises(ValueError): validate(df)
def test_clean():
    assert clean_text("  Ẹ káàárọ̀! 😃  ")=="Ẹ káàárọ̀! 😃"
    assert clean_text("hi @person a@b.com")=="hi [USER] [EMAIL]"
def test_duplicates(df):
    df.loc[1,"text"]=df.loc[0,"text"]
    groups=duplicate_groups(df); assert groups[0]==groups[1]
def test_annotations(df):
    a=pd.read_csv("data/samples/synthetic_annotations.csv")
    validate_annotations(a,df.text_id)
    assert fleiss_kappa(a)["fleiss_kappa"]==1
    with pytest.raises(ValueError): validate_annotations(a.iloc[1:],df.text_id)
    a.loc[:2,"sentiment_label"]=["positive","negative","neutral"]
    consensus,queue=adjudicate(a)
    assert consensus.iloc[0].sentiment=="" and len(queue)==3
def test_kappa():
    a=pd.DataFrame({"text_id":["a"]*3+["b"]*3,"sentiment_label":["positive","negative","neutral"]*2})
    assert fleiss_kappa(a)["fleiss_kappa"]==pytest.approx(-.5)
def test_split(df):
    df.loc[3,"text"]=df.loc[0,"text"]
    a=split_dataset(df); b=split_dataset(df)
    for key in a: pd.testing.assert_frame_equal(a[key],b[key])
    assert set(a["train"].duplicate_group).isdisjoint(a["test"].duplicate_group)
    assert sum(map(len,a.values()))==len(df)
def test_model(df,tmp_path):
    parts=split_dataset(df); model=train(parts["train"],{})
    pred=model.predict(parts["validation"].text)
    joblib.dump(model,tmp_path/"m.joblib")
    assert (pred==joblib.load(tmp_path/"m.joblib").predict(parts["validation"].text)).all()
    assert 0<=metrics(parts["validation"],pred)["macro_f1"]<=1
