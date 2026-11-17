from pathlib import Path

import pandas as pd
import pytest

from naija_sentiment.benchmark import build_svm_pipeline, clean_benchmark_text, prepare_frames, standardize_hf_split
from naija_sentiment.pilot import save_pilot_submission, validate_pilot_text
from naija_sentiment.prediction import predict_sentiment
from scripts.download_data import parquet_files_by_split


class FakeFeature:
    names = ["positive", "neutral", "negative"]

    def int2str(self, value):
        return self.names[value]


class FakeSplit:
    features = {"label": FakeFeature()}

    def to_pandas(self):
        return pd.DataFrame({"tweet": ["I dey happy", "This bad well well"], "label": [0, 2]})


def test_standardize_inspects_schema_and_decodes_labels():
    frame = standardize_hf_split(FakeSplit(), "pcm", "train")
    assert frame.columns.tolist() == ["text_id", "text", "sentiment", "source_language", "source_dataset", "original_split"]
    assert frame.sentiment.tolist() == ["positive", "negative"]
    assert frame.source_language.unique().tolist() == ["Nigerian Pidgin"]


def test_converted_parquet_file_selection_requires_every_split():
    files = [
        "pcm/train/0000.parquet",
        "pcm/validation/0000.parquet",
        "pcm/test/0000.parquet",
        "hau/train/0000.parquet",
    ]
    assert parquet_files_by_split(files, "pcm")["test"] == ["pcm/test/0000.parquet"]
    with pytest.raises(ValueError):
        parquet_files_by_split(files, "hau")


def test_conservative_cleaning_preserves_language_emoji_and_punctuation():
    value = "  Ẹ káàárọ̀! 😃  See https://example.org  @Somebody +234 801 234 5678 "
    assert clean_benchmark_text(value) == "Ẹ káàárọ̀! 😃 See [URL] @user [PHONE]"


def test_prepare_data_deduplicates_and_preserves_first_official_split():
    base = pd.DataFrame(
        {
            "text_id": ["a", "b", "c"],
            "text": ["Same text", "Same text", "Different"],
            "sentiment": ["positive", "positive", "neutral"],
            "source_language": ["Nigerian Pidgin"] * 3,
            "source_dataset": ["HausaNLP/NaijaSenti-Twitter"] * 3,
            "original_split": ["train", "test", "test"],
        }
    )
    prepared, report = prepare_frames([base])
    assert len(prepared) == 2
    assert prepared.loc[prepared.text.eq("Same text"), "original_split"].item() == "train"
    assert report["duplicates_removed"] == 1


def test_svm_pipeline_and_probability_prediction():
    texts = []
    labels = []
    phrases = {
        "positive": ["I love this good service", "this app make sense", "today sweet me well"],
        "negative": ["I hate this bad service", "network dey frustrate me", "this thing no good"],
        "neutral": ["meeting starts by noon", "the item is on table", "today is monday"],
    }
    for label, examples in phrases.items():
        for repeat in range(3):
            for example in examples:
                texts.append(f"{example} {repeat}")
                labels.append(label)
    model = build_svm_pipeline(calibrated=True)
    model.fit(texts, labels)
    result = predict_sentiment(model, "this app make sense")
    assert result["sentiment"] in {"positive", "negative", "neutral"}
    assert result["confidence"] is not None
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-6) == 1


def test_pilot_submission_requires_consent_and_blocks_identifiers(tmp_path: Path):
    with pytest.raises(ValueError):
        validate_pilot_text("Call me on +234 801 234 5678")
    destination = tmp_path / "pilot.csv"
    with pytest.raises(ValueError):
        save_pilot_submission(destination, text="Service no good", language_mix="Nigerian Pidgin", topic="telecoms", consent_confirmed=False, sentiment_suggestion="negative")
    identifier = save_pilot_submission(destination, text="Service no good", language_mix="Nigerian Pidgin", topic="telecoms", consent_confirmed=True, sentiment_suggestion="negative")
    saved = pd.read_csv(destination)
    assert saved.submission_id.item() == identifier
    assert saved.consent_confirmed.item()
