"""Streamlit MVP for the CTE 442 Nigerian multilingual sentiment research project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from naija_sentiment.pilot import save_pilot_submission
from naija_sentiment.prediction import load_pipeline, predict_sentiment

DATA_PATH = ROOT / "data/processed/naijasenti_combined.csv"
MODEL_PATH = ROOT / "artifacts/svm_sentiment_pipeline.joblib"
METRICS_PATH = ROOT / "artifacts/metrics.json"
REPORT_PATH = ROOT / "reports/classification_report.csv"
CONFUSION_PATH = ROOT / "reports/confusion_matrix.png"
PILOT_PATH = ROOT / "data/pilot/original_pilot_submissions.csv"

st.set_page_config(page_title="NaijaSentiment Research MVP", page_icon="🇳🇬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(145deg, #f4fbf7 0%, #ffffff 48%, #f7f4ff 100%); }
    [data-testid="stHeader"] { background: rgba(255,255,255,.75); }
    .hero { padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#075e36,#16875a); color: white; margin-bottom: 1rem; }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: .92; }
    .notice { padding: .9rem 1rem; border-left: 5px solid #7c3aed; background: #f3e8ff; border-radius: 8px; color: #3b1764; }
    .small-note { color: #4b5563; font-size: .9rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.88); border: 1px solid #dce8e1; padding: .8rem; border-radius: 14px; }
    </style>
    <div class="hero">
      <h1>Sentiment Analysis of Nigerian Pidgin and Code-Switched Text</h1>
      <p>CTE 442 · Group 4 · Yaba College of Technology</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    return pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False) if DATA_PATH.exists() else None


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_pipeline(MODEL_PATH)


def pilot_count() -> int:
    if not PILOT_PATH.exists():
        return 0
    return len(pd.read_csv(PILOT_PATH))


data = load_data()
metrics = load_metrics()

with st.sidebar:
    st.header("Prototype status")
    st.success("Dashboard ready")
    st.write("Benchmark data", "✅ Ready" if data is not None else "⏳ Run data scripts")
    st.write("SVM model", "✅ Ready" if MODEL_PATH.exists() else "⏳ Run training")
    st.write("Original pilot submissions", f"**{pilot_count()}**")
    st.caption("Public benchmark and original submissions are counted separately.")

overview, explorer, results_tab, analyser, pilot = st.tabs(
    ["Project overview", "Dataset explorer", "Model results", "Live analyser", "Original pilot data"]
)

with overview:
    st.subheader("Research question")
    st.write("Do multilingual transformer models handle Nigerian Pidgin and English–Yoruba or English–Hausa code-switching better than classical and recurrent models, and where do they fail?")
    st.markdown(
        '<div class="notice">“This prototype uses the public NaijaSenti dataset to establish the initial experimental pipeline. The original volunteer-contributed dataset required for the complete study is being collected separately and must not be confused with the public benchmark.”</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    columns = st.columns(4)
    columns[0].metric("Public benchmark records", f"{len(data):,}" if data is not None else "Not downloaded")
    columns[1].metric("Original submissions", f"{pilot_count():,}")
    columns[2].metric("Languages", f"{data.source_language.nunique()}" if data is not None else "3 planned")
    columns[3].metric("Sentiment classes", "3")
    st.markdown("**Benchmark:** `HausaNLP/NaijaSenti-Twitter` · Nigerian Pidgin primary; Hausa and Yoruba supporting datasets.")
    st.info("This MVP establishes a reproducible benchmark pipeline. It does not claim that NaijaSenti was collected by Group 4.")

with explorer:
    st.subheader("Public benchmark explorer")
    if data is None:
        st.warning("Benchmark data is not available yet. Run `python scripts/download_data.py` and `python scripts/prepare_data.py`.")
    else:
        filter_columns = st.columns([2, 2, 2, 3])
        language = filter_columns[0].selectbox("Language", ["All", *sorted(data.source_language.unique())])
        sentiment = filter_columns[1].selectbox("Sentiment", ["All", *sorted(data.sentiment.unique())])
        split = filter_columns[2].selectbox("Official split", ["All", "train", "validation", "test"])
        query = filter_columns[3].text_input("Search text", placeholder="Search anonymised benchmark text")
        filtered = data.copy()
        if language != "All": filtered = filtered.loc[filtered.source_language.eq(language)]
        if sentiment != "All": filtered = filtered.loc[filtered.sentiment.eq(sentiment)]
        if split != "All": filtered = filtered.loc[filtered.original_split.eq(split)]
        if query: filtered = filtered.loc[filtered.text.str.contains(query, case=False, na=False, regex=False)]
        st.caption(f"Showing {len(filtered):,} of {len(data):,} public benchmark rows. Identifiers are normalized during preparation.")
        st.dataframe(filtered[["text_id", "text", "sentiment", "source_language", "original_split"]].head(500), width="stretch", hide_index=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Class distribution**")
            counts = filtered.sentiment.value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
            fig, ax = plt.subplots(figsize=(6, 3.5)); sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette="viridis", ax=ax)
            ax.set(xlabel="Sentiment", ylabel="Records"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with right:
            st.markdown("**Language distribution**")
            counts = filtered.source_language.value_counts()
            fig, ax = plt.subplots(figsize=(6, 3.5)); sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False, palette="crest", ax=ax)
            ax.set(xlabel="Records", ylabel="Language"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with results_tab:
    st.subheader("Nigerian Pidgin SVM baseline")
    if metrics is None:
        st.warning("No verified model metrics exist yet. Run `python scripts/train_baseline.py` after preparing the official dataset.")
    else:
        test = metrics["test"]
        row = st.columns(5)
        row[0].metric("Test accuracy", f"{test['accuracy']:.3f}")
        row[1].metric("Test macro F1", f"{test['macro_f1']:.3f}")
        row[2].metric("Training time", f"{metrics['training_seconds']:.1f}s")
        row[3].metric("Test inference", f"{test['inference_seconds']:.2f}s")
        row[4].metric("Model size", f"{metrics['model_size_megabytes']:.1f} MB")
        st.caption(f"Trained on {metrics['number_of_training_records']:,} official Nigerian Pidgin training records. Test data was not used for training.")
        st.markdown("**Accuracy** is the proportion of correct predictions. **Macro F1** gives equal weight to each sentiment class. **Precision** asks how often a predicted class is correct; **recall** asks how many true examples of that class were found.")
        left, right = st.columns([1.15, 1])
        with left:
            if REPORT_PATH.exists(): st.dataframe(pd.read_csv(REPORT_PATH), width="stretch", hide_index=True)
        with right:
            if CONFUSION_PATH.exists(): st.image(str(CONFUSION_PATH), caption="Official Nigerian Pidgin test split")

with analyser:
    st.subheader("Live sentiment analyser")
    samples = [
        "This network dey frustrate person abeg.",
        "The app make sense, but una customer service no try.",
        "The service was good, sugbon price ti ga.",
    ]
    sample_columns = st.columns(3)
    for index, sample in enumerate(samples):
        if sample_columns[index].button(f"Use sample {index + 1}", width="stretch"):
            st.session_state["analysis_text"] = sample
    text = st.text_area("Enter Nigerian Pidgin or code-switched text", key="analysis_text", height=130)
    if st.button("Analyse sentiment", type="primary"):
        if not MODEL_PATH.exists():
            st.error("The trained model is unavailable. Run `python scripts/train_baseline.py` first.")
        else:
            try:
                prediction = predict_sentiment(cached_model(), text)
                st.success(f"Predicted sentiment: **{prediction['sentiment'].title()}**")
                if prediction["confidence"] is not None:
                    st.metric("Calibrated confidence", f"{prediction['confidence']:.1%}")
                    st.bar_chart(pd.Series(prediction["probabilities"], name="Probability"))
                st.caption("Model: word + character TF-IDF with calibrated Linear SVM")
            except ValueError as exc:
                st.error(str(exc))
    st.warning("This baseline may struggle with sarcasm, slang, spelling variation, mixed sentiment and unfamiliar code-switching. It does not reliably identify every language mixture.")

with pilot:
    st.subheader("Original volunteer-contributed pilot data")
    st.write("Contribute text only. Do not include names, phone numbers, email addresses, usernames, matric numbers, or other personal identifiers.")
    st.metric("Original submissions stored locally", pilot_count())
    with st.form("pilot_submission", clear_on_submit=True):
        pilot_text = st.text_area("Text", help="Enter a Nigerian-language or code-switched statement without personal identifiers.")
        language_mix = st.selectbox("Language mix", ["Nigerian Pidgin", "English–Pidgin", "English–Yoruba", "English–Hausa", "Other code-switching"])
        topic = st.text_input("Topic", placeholder="e.g. telecoms, transport, education")
        suggestion = st.selectbox("Optional sentiment suggestion", ["No suggestion", "positive", "negative", "neutral"])
        consent = st.checkbox("I voluntarily contribute this text for academic research and confirm it contains no personal identifiers.")
        submitted = st.form_submit_button("Save pilot submission", type="primary")
    if submitted:
        try:
            submission_id = save_pilot_submission(PILOT_PATH, text=pilot_text, language_mix=language_mix, topic=topic, consent_confirmed=consent, sentiment_suggestion=suggestion)
            st.success(f"Submission saved locally. Reference: {submission_id[:8]}")
            st.cache_data.clear()
        except ValueError as exc:
            st.error(str(exc))
