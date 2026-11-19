from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/pdf/CTE442_Sentiment_Analysis_MVP_Report.pdf"
METRICS = json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((ROOT / "reports/dataset_summary.json").read_text(encoding="utf-8"))
CONFUSION = ROOT / "reports/confusion_matrix.png"

GREEN = colors.HexColor("#087443")
DARK_GREEN = colors.HexColor("#064A30")
PURPLE = colors.HexColor("#6D28D9")
PALE_GREEN = colors.HexColor("#E8F3ED")
PALE_PURPLE = colors.HexColor("#F1EAFE")
INK = colors.HexColor("#17251F")
MUTED = colors.HexColor("#52645B")
LINE = colors.HexColor("#C9D8D0")
WHITE = colors.white

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=31, textColor=WHITE, alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontName="Helvetica", fontSize=12, leading=18, textColor=colors.HexColor("#DDF5E8")))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=DARK_GREEN, spaceBefore=8, spaceAfter=10))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=PURPLE, spaceBefore=8, spaceAfter=6))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.6, leading=14, textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.2, leading=11, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=9.4, leading=14, textColor=colors.HexColor("#35136B")))
styles.add(ParagraphStyle(name="CenterSmall", parent=styles["Smallx"], alignment=TA_CENTER))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def bullet(text):
    return Paragraph("- " + text, styles["Bodyx"])


def styled_table(rows, widths=None, header=True, font_size=8.5):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [WHITE, colors.HexColor("#F7FAF8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    table.setStyle(TableStyle(commands))
    return table


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.line(20 * mm, height - 15 * mm, width - 20 * mm, height - 15 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, height - 11.5 * mm, "CTE 442 - Nigerian Multilingual Sentiment Analysis")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def cover_block():
    content = [
        P("SENTIMENT ANALYSIS OF NIGERIAN PIDGIN AND CODE-SWITCHED TEXT", "CoverTitle"),
        P("CTE 442 Research Project - Working MVP Report", "CoverSub"),
        Spacer(1, 13 * mm),
        P("Yaba College of Technology", "CoverSub"),
        P("Group 4 - 13 Members", "CoverSub"),
        Spacer(1, 10 * mm),
        P("Verified implementation status: 19 November 2026", "CoverSub"),
    ]
    table = Table([[content]], colWidths=[170 * mm], rowHeights=[105 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 15 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 15 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 12 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12 * mm),
    ]))
    return table


story = [cover_block(), Spacer(1, 11 * mm)]
story += [
    P("Purpose", "H2x"),
    P("This report explains the implemented research prototype, how its data and model pipeline work, the verified baseline results, the limitations of the current dataset, and the steps needed to complete the broader academic study."),
    Table([[P("DATASET-USE DISCLAIMER", "Callout"), P("Only use data that the research team created, received with consent, or is permitted by a valid licence. The team remains responsible for privacy review, restrictions, attribution, secure storage, and correct citation. Loading data into the application does not transfer ownership or prove permission.", "Smallx")]], colWidths=[43 * mm, 120 * mm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_PURPLE),
        ("BOX", (0, 0), (-1, -1), 0.8, PURPLE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ])),
    PageBreak(),
]

story += [
    P("1. Project overview", "H1x"),
    P("Research question", "H2x"),
    P("Do multilingual transformer models handle Nigerian Pidgin and English-Yoruba or English-Hausa code-switching better than classical and recurrent models, and where do they fail?"),
    P("Current MVP scope", "H2x"),
    P("The current implementation is a complete first-stage experiment: dataset ingestion, conservative cleaning, leakage-aware split preservation, TF-IDF feature extraction, calibrated Linear SVM training, validation and test evaluation, saved artifacts, live predictions, and a five-tab Streamlit dashboard."),
    bullet("The working baseline is implemented and reproducible."),
    bullet("The dashboard can accept a team-supplied CSV after permission or licence confirmation."),
    bullet("fastText, BiLSTM, mBERT, AfroXLMR, and few-shot LLM comparisons remain future experiments."),
    P("What the current dataset can answer", "H2x"),
    P("The present data is sufficient for the MVP and for establishing a classical baseline. It is not sufficient by itself to answer the full research question because it does not explicitly encode the required English-Pidgin, English-Yoruba, and English-Hausa language-mix categories, and the Nigerian Pidgin training partition has very limited neutral coverage."),
]

story += [
    P("2. Dataset and preparation", "H1x"),
    P("Downloaded source counts", "H2x"),
    styled_table([
        ["Language", "Train", "Validation", "Test", "Total"],
        ["Nigerian Pidgin", "5,121", "1,281", "4,154", "10,556"],
        ["Hausa", "14,172", "2,677", "5,303", "22,152"],
        ["Yoruba", "8,522", "2,090", "4,515", "15,127"],
        ["Total", "27,815", "6,048", "13,972", "47,835"],
    ], widths=[48 * mm, 27 * mm, 29 * mm, 27 * mm, 29 * mm]),
    Spacer(1, 5 * mm),
    P("Cleaning outcome", "H2x"),
    styled_table([
        ["Measure", "Verified value"],
        ["Rows before cleaning", f"{SUMMARY['rows_before_cleaning']:,}"],
        ["Rows after cleaning", f"{SUMMARY['rows_after_cleaning']:,}"],
        ["Duplicates removed", f"{SUMMARY['duplicates_removed']:,}"],
        ["Missing values", f"{SUMMARY['missing_value_count']:,}"],
        ["Empty rows removed", f"{SUMMARY['empty_rows_removed']:,}"],
    ], widths=[82 * mm, 78 * mm]),
    Spacer(1, 5 * mm),
    KeepTogether([
        P("Prepared language distribution", "H2x"),
        styled_table([
            ["Language", "Rows", "Share of prepared data"],
            *[[name, f"{count:,}", f"{count / SUMMARY['rows_after_cleaning']:.1%}"] for name, count in SUMMARY["language_distribution"].items()],
        ], widths=[70 * mm, 40 * mm, 50 * mm]),
        P("Cleaning is conservative: URLs, handles, email addresses, phone numbers, and excessive whitespace are normalized; Nigerian vocabulary, spelling variation, punctuation, and emoji are preserved. Exact duplicate language-text pairs are removed while the earliest official split is retained."),
    ]),
]

flow = styled_table([
    ["1. Supply", "2. Validate", "3. Clean", "4. Train", "5. Evaluate", "6. Serve"],
], widths=[27 * mm] * 6, font_size=8.5)

story += [
    P("3. How the system works", "H1x"),
    flow,
    Spacer(1, 5 * mm),
    P("Stage 1 - Dataset supply and governance", "H2x"),
    P("A team-supplied CSV is uploaded through Dataset setup. The importer requires a dataset name, permission basis, and a licence, consent, source, or citation reference. The operator must confirm authorisation before the file is accepted. The importer supports text/tweet, sentiment/label, split, and an optional language column."),
    P("Stage 2 - Validation", "H2x"),
    P("The pipeline rejects empty files, missing required columns, ambiguous numeric sentiment labels, unsupported labels, and invalid split names. Sentiment must be positive, negative, or neutral. Split must be train, validation, or test. This prevents accidental label invention and preserves a defensible experimental design."),
    P("Stage 3 - Conservative cleaning", "H2x"),
    P("URLs are replaced with [URL], mentions with @user, and common email or phone patterns with placeholders. Extra whitespace is collapsed. The system does not translate, rewrite, stem, or standardize Nigerian spellings because these forms may carry sentiment and dialect information."),
    P("Stage 4 - Feature construction", "H2x"),
    P("Two TF-IDF feature spaces are built in parallel. Word unigrams and bigrams capture words and short phrases. Character 3- to 5-grams capture subword patterns, spelling variation, affixes, and informal forms that are common in Nigerian Pidgin and code-switched text. The two sparse matrices are joined with a FeatureUnion."),
    P("Stage 5 - Model training", "H2x"),
    P("A class-weighted Linear Support Vector Classifier learns a separating boundary for positive, negative, and neutral classes. CalibratedClassifierCV fits sigmoid calibration using training folds only, enabling the dashboard to display probability-like confidence values. The complete cleaner, TF-IDF feature union, calibration layer, and classifier are saved together with joblib."),
    P("Stage 6 - Evaluation", "H2x"),
    P("Model selection and diagnostic work use validation data. The held-out test split is evaluated separately. The pipeline records accuracy, macro F1, per-class precision, recall and F1, confusion matrix, training and inference times, model size, training count, random seed, and hardware."),
    PageBreak(),
]

story += [
    P("4. Baseline experiment", "H1x"),
    P("Configuration", "H2x"),
    styled_table([
        ["Item", "Value"],
        ["Model", METRICS["model"]],
        ["Primary language", METRICS["primary_language"]],
        ["Training records", f"{METRICS['number_of_training_records']:,}"],
        ["Random seed", str(METRICS["random_seed"])],
        ["Training time", f"{METRICS['training_seconds']:.2f} seconds"],
        ["Model size", f"{METRICS['model_size_megabytes']:.2f} MB"],
        ["Hardware", METRICS["hardware"]],
    ], widths=[50 * mm, 110 * mm]),
    Spacer(1, 6 * mm),
    P("Overall results", "H2x"),
    styled_table([
        ["Split", "Records", "Accuracy", "Macro F1", "Inference time"],
        ["Validation", f"{METRICS['validation']['records']:,}", f"{METRICS['validation']['accuracy']:.4f}", f"{METRICS['validation']['macro_f1']:.4f}", f"{METRICS['validation']['inference_seconds']:.3f} s"],
        ["Test", f"{METRICS['test']['records']:,}", f"{METRICS['test']['accuracy']:.4f}", f"{METRICS['test']['macro_f1']:.4f}", f"{METRICS['test']['inference_seconds']:.3f} s"],
    ], widths=[34 * mm, 30 * mm, 31 * mm, 31 * mm, 34 * mm]),
    Spacer(1, 6 * mm),
    P("Test results by class", "H2x"),
    styled_table([
        ["Class", "Precision", "Recall", "F1", "Support"],
        *[[label.title(), f"{METRICS['test']['classification_report'][label]['precision']:.4f}", f"{METRICS['test']['classification_report'][label]['recall']:.4f}", f"{METRICS['test']['classification_report'][label]['f1-score']:.4f}", f"{int(METRICS['test']['classification_report'][label]['support']):,}"] for label in ["negative", "neutral", "positive"]],
    ], widths=[40 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm]),
    P("Accuracy measures the total proportion of correct predictions. Macro F1 averages class-level F1 scores equally, making it more revealing when class performance is uneven. Precision measures how often a predicted class is correct; recall measures how many true examples of that class are found."),
]

if CONFUSION.exists():
    story += [PageBreak(), P("5. Confusion matrix and interpretation", "H1x"), Image(str(CONFUSION), width=155 * mm, height=111 * mm), P("Figure: Confusion matrix for the saved Nigerian Pidgin test experiment.", "CenterSmall")]

story += [
    P("Key finding", "H2x"),
    P("The baseline achieves 57.44% test accuracy but only 34.76% macro F1. It correctly identifies most negative items, but predicts no neutral test items correctly. The deduplicated training partition contains 3,003 negative, 1,540 positive, and only 66 neutral records. This extreme scarcity limits the classifier and shows why accuracy must not be reported alone."),
    P("What this means", "H2x"),
    bullet("The MVP proves that the complete technical pipeline works."),
    bullet("The baseline is not reliable enough for high-stakes or individual-level decisions."),
    bullet("Neutral data coverage and explicit code-switch labeling are higher priorities than superficial model tuning."),
    bullet("The official test split must remain untouched while improvements are selected using validation data."),
    PageBreak(),
]

story += [
    Spacer(1, 8 * mm),
    P("6. Dashboard and operating workflow", "H1x"),
    styled_table([
        ["Dashboard page", "Purpose"],
        ["Project overview", "Research question, dataset disclaimer, active source, record count, languages, classes, and splits."],
        ["Dataset explorer", "Searchable local table, language/sentiment/split filters, and class/language charts."],
        ["Model results", "Accuracy, macro F1, runtime, model size, per-class report, and confusion matrix."],
        ["Live analyser", "One-text sentiment prediction with calibrated confidence and explicit limitations."],
        ["Dataset setup", "Permission-gated import of a team-supplied CSV and local provenance record."],
    ], widths=[45 * mm, 115 * mm]),
    Spacer(1, 7 * mm),
    P("How live prediction works", "H2x"),
    P("The Streamlit application loads the saved pipeline. User text passes through the same cleaning and TF-IDF transformations used during training, then the calibrated SVM returns a sentiment class and confidence distribution. The dashboard also exposes dataset filters, saved metrics, the confusion matrix, and dataset import controls."),
    P("Team datasets may be imported through Dataset setup and must include text, sentiment, and train/validation/test split columns."),
    P("Reproduce the current application", "H2x"),
    P("1. Activate the Python virtual environment.<br/>2. Install requirements.<br/>3. Prepare or import an authorised dataset.<br/>4. Train the baseline.<br/>5. Run the tests.<br/>6. Launch Streamlit and open the local URL."),
    Table([[P("source .venv/bin/activate<br/>pip install -r requirements.txt<br/>python scripts/prepare_data.py<br/>python scripts/train_baseline.py<br/>pytest -q<br/>streamlit run app.py", "Smallx")]], colWidths=[160 * mm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF5F1")),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ])),
    PageBreak(),
]

story += [
    P("7. Limitations and next research stage", "H1x"),
    bullet("Current source-language values do not directly identify English-Pidgin, English-Yoruba, or English-Hausa mixtures."),
    bullet("Neutral Nigerian Pidgin training support is too small for reliable three-class modeling."),
    bullet("Sarcasm, slang, spelling variation, dialect, and mixed sentiment are not yet explicitly tagged."),
    bullet("Only the classical SVM baseline is complete; comparative models remain unimplemented or unevaluated."),
    bullet("Calibrated confidence is not a guarantee of correctness and should be interpreted cautiously."),
    P("Recommended next actions", "H2x"),
    P("1. Select the final team dataset and preserve its permission or licence record."),
    P("2. Add explicit language-mix labels and improve neutral coverage."),
    P("3. Use three independent annotators where the team creates labels, calculate Fleiss' kappa, and adjudicate disagreements."),
    P("4. Freeze train, validation, and test partitions before comparative experiments."),
    P("5. Implement fastText and BiLSTM, then mBERT and AfroXLMR, using identical splits and repeated seeds."),
    P("6. Run a controlled few-shot LLM benchmark only when dataset permissions allow external processing."),
    P("7. Produce learning curves, uncertainty estimates, and qualitative error/disparity analysis."),
    P("Conclusion", "H2x"),
    P("The project has a functioning, tested MVP and a defensible first baseline. Its most important result is not the headline accuracy; it is the evidence that data composition, especially neutral-class scarcity and missing language-mix labels, currently prevents the baseline from fully answering the research question. The next phase should improve dataset coverage and annotation quality before comparing larger models."),
]

story += [
    P("References", "H1x"),
    P("Muhammad, S. H., Adelani, D. I., Ruder, S., Ahmad, I. S., Abdulmumin, I., Bello, B. S., Choudhury, M., Emezue, C. C., Abdullahi, S. S., Aremu, A., Jeorge, A., and Brazdil, P. (2022). NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis. Proceedings of LREC 2022, 590-602."),
    P("Dataset repository: https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter"),
    P("Paper: https://aclanthology.org/2022.lrec-1.63/"),
    P("Project implementation sources: app.py; scripts/download_data.py; scripts/prepare_data.py; scripts/train_baseline.py; src/naija_sentiment/benchmark.py; src/naija_sentiment/manual_data.py; artifacts/metrics.json; reports/dataset_summary.json."),
    Spacer(1, 8 * mm),
    Table([[P("Verification record", "Callout"), P("39 automated tests passed. Streamlit application smoke test: five tabs, zero runtime exceptions. Model, metrics, classification report, test predictions, and confusion matrix were regenerated from the prepared dataset.", "Smallx")]], colWidths=[45 * mm, 115 * mm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
        ("BOX", (0, 0), (-1, -1), 0.8, GREEN),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ])),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=20 * mm,
    leftMargin=20 * mm,
    topMargin=21 * mm,
    bottomMargin=17 * mm,
    title="CTE 442 Sentiment Analysis of Nigerian Pidgin and Code-Switched Text",
    author="Group 4, Yaba College of Technology",
    subject="Research MVP implementation report",
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(OUTPUT)
