"""
Shared pytest setup.

model_development.py reads LAB2_WORKING_DIR / LAB2_MODEL_DIR when it is first
imported, so we point them at a temporary folder *before* anything imports it.
That way the tests never overwrite the real files in working_data/ and model/.
"""
import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="lab2_tests_")
os.environ["LAB2_WORKING_DIR"] = os.path.join(_TMP, "working_data")
os.environ["LAB2_MODEL_DIR"] = os.path.join(_TMP, "model")

# Same import path Airflow uses: the dags/ folder is on sys.path.
DAGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dags")
sys.path.insert(0, DAGS_DIR)

import pickle  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import model_development as md  # noqa: E402


@pytest.fixture(scope="session")
def raw_path():
    """The real diabetes CSV, loaded once and saved as a pickle."""
    return md.load_data()


@pytest.fixture(scope="session")
def prep_path(raw_path):
    """Preprocessed train/test split, shared by the model tests."""
    return md.data_preprocessing(raw_path, test_size=0.3)


@pytest.fixture
def write_df(tmp_path):
    """Save a dataframe as a pickle and return its path (for bad-data tests)."""
    def _write(df: pd.DataFrame) -> str:
        path = tmp_path / "df.pkl"
        with open(path, "wb") as f:
            pickle.dump(df, f)
        return str(path)
    return _write
