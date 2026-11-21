"""Streamlit MVP for the CTE 442 Nigerian multilingual sentiment research project."""

from __future__ import annotations

import json
import sys
import time
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
    .hero { padding: 1.45rem 1.65rem; border-radius: 20px; background: linear-gradient(120deg,#075e36,#16875a); color: white; margin-bottom: 1rem; box-shadow: 0 12px 28px rgba(7,94,54,.14); }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: .92; }
    .notice { padding: .9rem 1rem; border-left: 5px solid #7c3aed; background: #f3e8ff; border-radius: 8px; color: #3b1764; }
    .guide-card { min-height: 120px; padding: 1rem; border: 1px solid #dce8e1; border-radius: 14px; background: rgba(255,255,255,.82); }
    .guide-card strong { color: #075e36; display: block; margin-bottom: .3rem; }
    .flow-card { padding: .8rem .85rem; border-left: 4px solid #16875a; border-radius: 0 10px 10px 0; background: #f0f8f3; margin: .5rem 0; }
    .pulse { display: inline-block; width: .65rem; height: .65rem; margin-right: .35rem; border-radius: 50%; background: #22c55e; animation: pulse 1.5s ease-out infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,.52); } 75% { box-shadow: 0 0 0 9px rgba(34,197,94,0); } 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
    .small-note { color: #4b5563; font-size: .9rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.88); border: 1px solid #dce8e1; padding: .8rem; border-radius: 14px; }
    button[kind="primary"] { border-radius: 10px; }
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


def source_summary(frame: pd.DataFrame) -> str:
    """Return a concise, human-readable description of active data sources."""
    return ", ".join(sorted(frame.source_dataset.astype(str).unique()))


def render_process_step(number: int, title: str, description: str) -> None:
    """Render one compact explanation of a real pipeline stage."""
    st.markdown(
        f'<div class="flow-card"><strong>{number}. {title}</strong><br/>{description}</div>',
        unsafe_allow_html=True,
    )


DATA_PATH = MANUAL_DATA_PATH if MANUAL_DATA_PATH.exists() else BENCHMARK_DATA_PATH
data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns) if DATA_PATH.exists() else None
metrics = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime_ns) if METRICS_PATH.exists() else None
using_manual_data = DATA_PATH == MANUAL_DATA_PATH

with st.sidebar:
    st.header("Research workspace")
    st.markdown('<span class="pulse"></span><strong>Interactive MVP running</strong>', unsafe_allow_html=True)
    st.write("Research dataset", "✅ Ready" if data is not None else "⏳ Import a CSV")
    st.write("SVM baseline", "✅ Ready" if MODEL_PATH.exists() else "⏳ Run training")
    st.write("Comparative models", "⏳ Planned")
    st.caption("Active source: team-supplied CSV" if using_manual_data else "Active source: prepared local dataset")
    st.divider()
    st.caption("Suggested flow")
    st.markdown("1. Understand the scope\n2. Explore the data\n3. Read the results\n4. Try a live prediction\n5. Import approved team data")

overview, explorer, results_tab, analyser, dataset_setup = st.tabs(
    ["Project overview", "Dataset explorer", "Model results", "Live analyser", "Dataset setup"]
)

with overview:
    st.subheader("What this project is testing")
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
        st.markdown("**Active dataset source:** " + source_summary(data))

    st.subheader("Where the project stands")
    progress_col, explanation_col = st.columns([1, 2])
    with progress_col:
        st.progress(0.45, text="MVP foundation: 45% of the full comparative study")
    with explanation_col:
        st.info("The data workflow, annotation infrastructure, reproducible SVM baseline, evaluation outputs, and interactive dashboard are working. The full research aim requires language-mix labels, three-person annotation of team data, and fair experiments with fastText, BiLSTM, mBERT, AfroXLMR, and a controlled few-shot LLM.")

    st.subheader("How to use this simulation")
    guide = st.columns(4)
    cards = [
        ("Explore", "Filter the active dataset and watch the charts update."),
        ("Interpret", "Compare accuracy with macro F1 before drawing a conclusion."),
        ("Try", "Enter text and observe the saved baseline's real prediction flow."),
        ("Extend", "Import only approved team data, then retrain reproducibly."),
    ]
    for column, (title, body) in zip(guide, cards):
        column.markdown(f'<div class="guide-card"><strong>{title}</strong>{body}</div>', unsafe_allow_html=True)

    with st.expander("Can the current MVP fully answer the research question?", expanded=True):
        st.warning("Not yet. It establishes a tested baseline and a research-safe workflow, but it cannot prove whether multilingual transformers are better until those models are trained on the same fixed splits. The current data also does not fully label the target English–Pidgin, English–Yoruba, and English–Hausa mixes.")
        st.markdown("**What is already defensible:** the dataset pipeline, privacy/permission checks, reproducible baseline, metrics, and live demonstration.  \n**What must be completed:** balanced team data, independent annotation and agreement reporting, language-mix analysis, comparative models, learning curves, and error/bias analysis.")

with explorer:
    st.subheader("Research dataset explorer")
    st.caption("Adjust any filter below. The records, examples, and charts respond to your current view.")
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
        summary = st.columns(4)
        summary[0].metric("Visible rows", f"{len(filtered):,}")
        summary[1].metric("Positive", f"{int(filtered.sentiment.eq('positive').sum()):,}")
        summary[2].metric("Negative", f"{int(filtered.sentiment.eq('negative').sum()):,}")
        summary[3].metric("Neutral", f"{int(filtered.sentiment.eq('neutral').sum()):,}")
        st.caption(f"Showing {len(filtered):,} of {len(data):,} locally prepared rows. Common identifiers are normalized during preparation.")
        if filtered.empty:
            st.info("No records match this combination. Clear a filter or try a shorter search phrase.")
        else:
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
        with st.expander("How to read this tab"):
            st.markdown("- **Filters** let you inspect a subgroup without changing the saved dataset.\n- **Class distribution** reveals whether sentiment labels are balanced. A low-count class can make accuracy look stronger than it is.\n- **Language distribution** shows which source languages are represented. Source language is not automatically the same as a code-switch label.\n- Only anonymised/local prepared text is displayed; do not use this view to recover or share personal information.")

with results_tab:
    st.subheader("Saved SVM baseline")
    st.caption("These values are loaded from the saved experiment artifact; they are not demo numbers.")
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
        if test["macro_f1"] + 0.15 < test["accuracy"]:
            st.warning("Read the two headline metrics together: macro F1 is materially lower than accuracy, which signals uneven behaviour across sentiment classes. Inspect the class report and confusion matrix before treating the model as reliable.")
        result_explainer, report_view, matrix_view = st.tabs(["What these results mean", "Class report", "Confusion matrix"])
        with result_explainer:
            st.markdown("**Accuracy** is the proportion of all correct predictions. **Macro F1** gives the same importance to each class, so it exposes weak performance on a minority class. **Precision** asks whether a predicted class is usually right; **recall** asks whether the model finds most true items in that class.")
            render_process_step(1, "Train", "Word and character TF-IDF turns the approved training text into numerical features.")
            render_process_step(2, "Validate", "Configuration decisions use validation data rather than the final test set.")
            render_process_step(3, "Test once", "The held-out official test split produces the reported metrics and matrix.")
        with report_view:
            if REPORT_PATH.exists():
                st.dataframe(pd.read_csv(REPORT_PATH), width="stretch", hide_index=True)
            else:
                st.info("The classification report file is not available yet.")
        with matrix_view:
            if CONFUSION_PATH.exists():
                st.image(str(CONFUSION_PATH), caption="Rows are actual sentiment; columns are predicted sentiment.")
            else:
                st.info("The confusion matrix will appear after baseline training.")

with analyser:
    st.subheader("Live sentiment analyser")
    st.caption("This is a live run of the saved pipeline—not a hand-written demonstration. It explains each real stage without pretending that the model understands intent like a person.")
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
    if st.button("Analyse sentiment", type="primary", use_container_width=True):
        if not MODEL_PATH.exists():
            st.error("The trained model is unavailable. Run `python scripts/train_baseline.py` first.")
        else:
            try:
                trace = st.status("Preparing your text for the saved baseline…", expanded=True)
                progress = st.progress(5, text="Step 1 of 3: checking the input")
                time.sleep(0.15)
                trace.write("✓ Input accepted. URLs, mentions, and extra spaces are normalised by the same preprocessor used in training.")
                progress.progress(35, text="Step 2 of 3: creating word and character features")
                model = cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
                time.sleep(0.15)
                trace.write("✓ Word n-grams capture short phrases; character n-grams help with spelling variation and informal forms.")
                progress.progress(70, text="Step 3 of 3: applying the calibrated Linear SVM")
                prediction = predict_sentiment(model, text)
                trace.write("✓ The classifier selected its highest-scoring sentiment class. Calibration provides a probability-like confidence value.")
                progress.progress(100, text="Analysis complete")
                trace.update(label="Live sentiment analysis complete", state="complete", expanded=False)
                st.success(f"Predicted sentiment: **{prediction['sentiment'].title()}**")
                if prediction["confidence"] is not None:
                    st.metric("Calibrated confidence", f"{prediction['confidence']:.1%}")
                    st.bar_chart(pd.Series(prediction["probabilities"], name="Probability"))
                st.markdown("**What the result means:** the prediction is the label selected by the saved word-and-character TF-IDF Linear SVM. Confidence describes the model's separation of the available classes; it is not a guarantee that the interpretation is correct.")
            except ValueError as exc:
                st.error(str(exc))
    with st.expander("What the analyser can and cannot do"):
        st.markdown("**It can:** apply the exact saved baseline consistently to new, short text and show its calibrated score distribution.  \n**It cannot:** reliably detect sarcasm, resolve mixed sentiment, explain a specific prediction causally, or identify every language mixture. Treat it as a research demonstration, not an automated decision-maker.")
    st.warning("This baseline may struggle with sarcasm, slang, spelling variation, mixed sentiment and unfamiliar code-switching. It does not reliably identify every language mixture.")

with dataset_setup:
    st.subheader("Use a team-supplied dataset")
    st.markdown(
        '<div class="notice"><strong>Required confirmation:</strong> Upload only a CSV the team is authorised to use. You remain responsible for consent or licensing, privacy review, restrictions, attribution, and secure storage.</div>',
        unsafe_allow_html=True,
    )
    st.write("Required columns: `text` or `tweet`; `sentiment` or `label`; and `split` with train, validation, or test. Optional language column: `source_language`, `language`, or `language_mix`.")
    upload_guide, privacy_guide = st.columns(2)
    upload_guide.info("**Import flow:** upload → schema check → label/split validation → conservative cleaning → local processed CSV and provenance record.")
    privacy_guide.warning("**Before uploading:** remove names, handles, phone numbers, emails, and any direct identifiers. Keep consent/licence evidence outside the public repository.")
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
                st.json({"rows_received": result["provenance"]["rows_received"], "rows_retained": result["provenance"]["rows_retained"], "permission_basis": result["provenance"]["permission_basis"]})
                st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
