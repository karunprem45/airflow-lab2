"""Unit tests for the plain-Python ML steps in dags/src/model_development.py."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from src import model_development as md


# ---------- load_data ----------
def test_load_data_reads_the_diabetes_csv(raw_path):
    with open(raw_path, "rb") as f:
        df = pickle.load(f)
    assert df.shape == (768, 9)
    assert list(df.columns) == md.NUM_COLUMNS + [md.TARGET]


# ---------- validate_data ----------
def test_validate_data_passes_on_real_data(raw_path):
    report = md.validate_data(raw_path)
    assert report["rows"] == 768
    assert 0.3 < report["positive_rate"] < 0.4  # about 35% have diabetes


def _good_df(rows=150):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(1, 100, size=(rows, len(md.NUM_COLUMNS))), columns=md.NUM_COLUMNS)
    df[md.TARGET] = [0, 1] * (rows // 2)
    return df


@pytest.mark.parametrize(
    "break_data, message",
    [
        (lambda df: df.drop(columns=["Glucose"]), "Missing required columns"),
        (lambda df: df.head(50), "need at least"),
        (lambda df: df.assign(BMI=np.nan), "null values"),
        (lambda df: df.assign(Outcome=2), "binary"),
    ],
    ids=["missing-column", "too-few-rows", "nulls", "non-binary-target"],
)
def test_validate_data_rejects_bad_data(write_df, break_data, message):
    path = write_df(break_data(_good_df()))
    with pytest.raises(ValueError, match=message):
        md.validate_data(path)


# ---------- data_preprocessing ----------
def test_preprocessing_splits_70_30_and_stratifies(prep_path):
    with open(prep_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    assert len(X_train) + len(X_test) == 768
    assert len(X_test) == pytest.approx(768 * 0.3, abs=1)
    # Stratified: both splits keep roughly the same diabetes rate.
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)


def test_preprocessing_fills_impossible_zeros(prep_path):
    with open(prep_path, "rb") as f:
        X_train, X_test, _, _ = pickle.load(f)
    # Zeros became NaN and were imputed, so no NaN may reach the models.
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    # Standardized training features have mean ~0.
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)


# ---------- train_and_evaluate ----------
@pytest.mark.parametrize("model_name", list(md.MODEL_FACTORIES))
def test_each_candidate_trains_and_beats_baseline(prep_path, model_name):
    result = md.train_and_evaluate(prep_path, model_name)
    assert result["model_name"] == model_name
    assert os.path.exists(result["model_path"])
    # Always guessing "no diabetes" scores about 0.65; a real model must do better.
    assert 0.65 < result["accuracy"] <= 1
    assert 0 < result["f1"] <= 1


# ---------- select_best_model ----------
def test_select_best_ranks_by_accuracy_then_f1():
    results = [
        {"model_name": "a", "accuracy": 0.74, "f1": 0.60, "model_path": "a"},
        {"model_name": "b", "accuracy": 0.75, "f1": 0.55, "model_path": "b"},
        {"model_name": "c", "accuracy": 0.75, "f1": 0.58, "model_path": "c"},
    ]
    best = md.select_best_model(results, run_id="test_run")
    assert best["model_name"] == "c"  # tie on accuracy, higher F1 wins

    with open(md.LEADERBOARD_PATH) as f:
        board = json.load(f)
    assert board["run_id"] == "test_run"
    assert [m["model_name"] for m in board["models"]] == ["c", "b", "a"]


# ---------- promote_model + score_production_model ----------
def test_promoted_model_reloads_with_the_same_score(prep_path):
    best = md.train_and_evaluate(prep_path, "random_forest")
    metadata = md.promote_model(best, run_id="test_run", min_accuracy=0.72)

    assert os.path.exists(md.PRODUCTION_MODEL)
    with open(md.PRODUCTION_METADATA) as f:
        assert json.load(f) == metadata
    assert metadata["promoted_by_run"] == "test_run"
    assert md.score_production_model(prep_path) == best["accuracy"]
