# File: src/model_development.py
"""
Plain-Python ML helpers used by the Airflow_Lab2 DAG.

Nothing in this module imports Airflow: every function takes and returns
simple values (file paths, dicts, floats) so the DAG can pass them between
tasks through XCom and each step can be tested on its own.
"""
import json
import os
import pickle
import shutil
from datetime import datetime, timezone

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

WORKING_DIR = os.getenv("LAB2_WORKING_DIR", "/opt/airflow/working_data")
MODEL_DIR = os.getenv("LAB2_MODEL_DIR", "/opt/airflow/model")
CANDIDATE_DIR = os.path.join(MODEL_DIR, "candidates")
PRODUCTION_DIR = os.path.join(MODEL_DIR, "production")
PRODUCTION_MODEL = os.path.join(PRODUCTION_DIR, "model.sav")
PRODUCTION_METADATA = os.path.join(PRODUCTION_DIR, "metadata.json")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

for _d in (WORKING_DIR, CANDIDATE_DIR, PRODUCTION_DIR):
    os.makedirs(_d, exist_ok=True)

DROP_COLUMNS = ["Timestamp", "Clicked on Ad", "Ad Topic Line", "Country", "City"]
NUM_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
TARGET = "Clicked on Ad"
REQUIRED_COLUMNS = NUM_COLUMNS + [TARGET]

# Candidate models. The DAG creates one mapped task per key at runtime.
MODEL_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
}


def load_data() -> str:
    """Load the CSV and persist the raw dataframe to a pickle. Returns its path."""
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "advertising.csv"
    )
    df = pd.read_csv(csv_path)

    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path


def validate_data(file_path: str, min_rows: int = 100) -> dict:
    """
    Data-quality checks. Raises ValueError if the data is unusable (the DAG
    turns that into a hard failure with no retries). Returns a small report.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) < min_rows:
        raise ValueError(f"Only {len(df)} rows, need at least {min_rows}")
    nulls = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    if nulls:
        raise ValueError(f"Found {nulls} null values in required columns")
    labels = sorted(df[TARGET].unique().tolist())
    if labels != [0, 1]:
        raise ValueError(f"Target must be binary 0/1, got {labels}")

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "positive_rate": round(float(df[TARGET].mean()), 4),
    }


def data_preprocessing(file_path: str, test_size: float = 0.3) -> str:
    """Split, scale, and save (X_train, X_test, y_train, y_test). Returns path."""
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    X = df.drop(DROP_COLUMNS, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    ct = make_column_transformer(
        (MinMaxScaler(), NUM_COLUMNS),
        (StandardScaler(), NUM_COLUMNS),
        remainder="passthrough",
    )
    X_train_tr = ct.fit_transform(X_train)
    X_test_tr = ct.transform(X_test)

    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train_tr, X_test_tr, y_train.values, y_test.values), f)
    return out_path


def train_and_evaluate(file_path: str, model_name: str) -> dict:
    """Train one candidate model, save it, and return its test metrics."""
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model = MODEL_FACTORIES[model_name]()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    model_path = os.path.join(CANDIDATE_DIR, f"{model_name}.sav")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "model_path": model_path,
    }


def select_best_model(results: list[dict], run_id: str) -> dict:
    """Rank candidates by accuracy (F1 breaks ties) and write a leaderboard file."""
    ranked = sorted(results, key=lambda r: (r["accuracy"], r["f1"]), reverse=True)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump({"run_id": run_id, "models": ranked}, f, indent=2)
    return ranked[0]


def promote_model(best: dict, run_id: str, min_accuracy: float) -> dict:
    """Copy the winning candidate into production/ and write its metadata."""
    shutil.copyfile(best["model_path"], PRODUCTION_MODEL)
    metadata = {
        "model_name": best["model_name"],
        "accuracy": best["accuracy"],
        "f1": best["f1"],
        "min_accuracy": min_accuracy,
        "promoted_by_run": run_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(PRODUCTION_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def score_production_model(file_path: str) -> float:
    """Reload the production model from disk and re-score it (smoke test)."""
    with open(file_path, "rb") as f:
        _, X_test, _, y_test = pickle.load(f)
    with open(PRODUCTION_MODEL, "rb") as f:
        model = pickle.load(f)
    return round(float(model.score(X_test, y_test)), 4)
