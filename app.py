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

from naija_sentiment.manual_data import import_manual_dataset
from naija_sentiment.prediction import load_pipeline, predict_sentiment

BENCHMARK_DATA_PATH = ROOT / "data/processed/naijasenti_combined.csv"
MANUAL_DATA_PATH = ROOT / "data/processed/research_dataset.csv"
MANUAL_PROVENANCE_PATH = ROOT / "data/manual/dataset_provenance.json"
MANUAL_REPORT_PATH = ROOT / "reports/manual_dataset_summary.json"
MODEL_PATH = ROOT / "artifacts/svm_sentiment_pipeline.joblib"
METRICS_PATH = ROOT / "artifacts/metrics.json"
REPORT_PATH = ROOT / "reports/classification_report.csv"
CONFUSION_PATH = ROOT / "reports/confusion_matrix.png"

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
def load_data(path: str, modified_ns: int) -> pd.DataFrame:
    """Load processed data; the mtime argument invalidates stale Streamlit caches."""
    del modified_ns
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@st.cache_data(show_spinner=False)
def load_metrics(path: str, modified_ns: int) -> dict:
    """Load metrics and refresh automatically after retraining."""
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def cached_model(path: str, modified_ns: int):
    """Load the latest saved pipeline and invalidate after retraining."""
    del modified_ns
    return load_pipeline(path)


DATA_PATH = MANUAL_DATA_PATH if MANUAL_DATA_PATH.exists() else BENCHMARK_DATA_PATH
data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns) if DATA_PATH.exists() else None
metrics = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime_ns) if METRICS_PATH.exists() else None
using_manual_data = DATA_PATH == MANUAL_DATA_PATH

with st.sidebar:
    st.header("Prototype status")
    st.success("Dashboard ready")
    st.write("Research dataset", "✅ Ready" if data is not None else "⏳ Import a CSV")
    st.write("SVM model", "✅ Ready" if MODEL_PATH.exists() else "⏳ Run training")
    st.caption("Active source: team-supplied CSV" if using_manual_data else "Active source: prepared local dataset")

overview, explorer, results_tab, analyser, dataset_setup = st.tabs(
    ["Project overview", "Dataset explorer", "Model results", "Live analyser", "Dataset setup"]
)

with overview:
    st.subheader("Research question")
    st.write("Do multilingual transformer models handle Nigerian Pidgin and English–Yoruba or English–Hausa code-switching better than classical and recurrent models, and where do they fail?")
    st.markdown(
        '<div class="notice"><strong>Dataset-use disclaimer:</strong> Only use data that the research team created, received with consent, or is permitted by a valid licence. The team must remove personal identifiers, respect all restrictions, and provide the correct citation. Loading a dataset into this application does not transfer ownership or prove permission.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    columns = st.columns(4)
    columns[0].metric("Dataset records", f"{len(data):,}" if data is not None else "Not loaded")
    columns[1].metric("Languages", f"{data.source_language.nunique()}" if data is not None else "—")
    columns[2].metric("Sentiment classes", f"{data.sentiment.nunique()}" if data is not None else "—")
    columns[3].metric("Official splits", f"{data.original_split.nunique()}" if data is not None else "—")
    if data is not None:
        st.markdown("**Active dataset source:** " + ", ".join(sorted(data.source_dataset.astype(str).unique())))

with explorer:
    st.subheader("Research dataset explorer")
    if data is None:
        if (ROOT / "data/raw/download_manifest.json").exists():
            st.warning("Raw data is available but not prepared. Run `python scripts/prepare_data.py`, or import the team CSV in Dataset setup.")
        else:
            st.warning("No research dataset is active. Import a permitted CSV in Dataset setup.")
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
        st.caption(f"Showing {len(filtered):,} of {len(data):,} locally prepared rows. Common identifiers are normalized during preparation.")
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
    st.subheader("Saved SVM baseline")
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
        st.caption(f"Trained on {metrics['number_of_training_records']:,} training records from {metrics['source_dataset']}. Test data was not used for training.")
        st.markdown("**Accuracy** is the proportion of correct predictions. **Macro F1** gives equal weight to each sentiment class. **Precision** asks how often a predicted class is correct; **recall** asks how many true examples of that class were found.")
        left, right = st.columns([1.15, 1])
        with left:
            if REPORT_PATH.exists(): st.dataframe(pd.read_csv(REPORT_PATH), width="stretch", hide_index=True)
        with right:
            if CONFUSION_PATH.exists(): st.image(str(CONFUSION_PATH), caption="Saved model test split")

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
                prediction = predict_sentiment(cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns), text)
                st.success(f"Predicted sentiment: **{prediction['sentiment'].title()}**")
                if prediction["confidence"] is not None:
                    st.metric("Calibrated confidence", f"{prediction['confidence']:.1%}")
                    st.bar_chart(pd.Series(prediction["probabilities"], name="Probability"))
                st.caption("Model: word + character TF-IDF with calibrated Linear SVM")
            except ValueError as exc:
                st.error(str(exc))
    st.warning("This baseline may struggle with sarcasm, slang, spelling variation, mixed sentiment and unfamiliar code-switching. It does not reliably identify every language mixture.")

with dataset_setup:
    st.subheader("Use a team-supplied dataset")
    st.markdown(
        '<div class="notice"><strong>Required confirmation:</strong> Upload only a CSV the team is authorised to use. You remain responsible for consent or licensing, privacy review, restrictions, attribution, and secure storage.</div>',
        unsafe_allow_html=True,
    )
    st.write("Required columns: `text` or `tweet`; `sentiment` or `label`; and `split` with train, validation, or test. Optional language column: `source_language`, `language`, or `language_mix`.")
    with st.form("manual_dataset_import", clear_on_submit=False):
        uploaded = st.file_uploader("Dataset CSV", type=["csv"])
        dataset_name = st.text_input("Dataset name or source", placeholder="e.g. Group 4 manually prepared dataset")
        permission_basis = st.selectbox("Permission basis", ["Select one", "Team-created", "Consent obtained", "Licensed dataset", "Public-domain dataset"])
        reference = st.text_input("Licence, consent record, source, or citation reference")
        confirmation = st.checkbox("I confirm that the team is authorised to use this dataset and will comply with its privacy, citation, and licence requirements.")
        imported = st.form_submit_button("Validate and use dataset", type="primary")
    if imported:
        if uploaded is None:
            st.error("Choose a CSV file")
        else:
            try:
                result = import_manual_dataset(
                    uploaded,
                    dataset_name=dataset_name,
                    permission_basis=permission_basis,
                    licence_or_reference=reference,
                    confirmed=confirmation,
                    output_path=MANUAL_DATA_PATH,
                    provenance_path=MANUAL_PROVENANCE_PATH,
                    report_path=MANUAL_REPORT_PATH,
                )
                st.success(f"Dataset accepted: {result['provenance']['rows_retained']:,} cleaned rows stored locally.")
                st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
