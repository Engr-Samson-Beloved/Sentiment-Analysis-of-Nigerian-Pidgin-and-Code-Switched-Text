from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from ..data.cleaning import clean_text
def train(df, config):
    if df.sentiment.nunique()<2: raise ValueError("Training requires two classes")
    model=Pipeline([("tfidf",TfidfVectorizer(preprocessor=clean_text,ngram_range=tuple(config.get("ngram_range",[1,2])))),("svm",LinearSVC(C=config.get("C",1),random_state=config.get("seed",42)))])
    return model.fit(df.text,df.sentiment)
