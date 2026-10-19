"""
DAG integrity tests: both DAGs import cleanly and are wired the way the README
describes. These need Airflow installed, so run them inside the containers.
"""
import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

from conftest import DAGS_DIR  # noqa: E402
from src.lab2_common import PRODUCTION_MODEL_ASSET  # noqa: E402


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(DAGS_DIR)


@pytest.fixture(scope="module")
def lab2(dagbag):
    return dagbag.dags["Airflow_Lab2"]


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}
    assert set(dagbag.dags) == {"Airflow_Lab2", "Airflow_Lab2_Flask"}


def test_pipeline_has_expected_tasks(lab2):
    assert set(lab2.task_ids) == {
        "task_using_linked_owner",
        "load_data",
        "validate_data",
        "preprocess",
        "list_candidates",
        "train_candidate",
        "select_best",
        "quality_gate",
        "promote_model",
        "smoke_test_production_model",
        "reject_model",
        "pipeline_summary",
    }


def test_training_uses_dynamic_task_mapping(lab2):
    assert type(lab2.get_task("train_candidate")).__name__ == "DecoratedMappedOperator"


def test_quality_gate_branches_to_promote_or_reject(lab2):
    gate = lab2.get_task("quality_gate")
    assert gate.downstream_task_ids == {"promote_model", "reject_model"}


@pytest.mark.parametrize(
    "accuracy, expected",
    [(0.7489, "promote_model"), (0.72, "promote_model"), (0.71, "reject_model")],
)
def test_quality_gate_decision(lab2, accuracy, expected):
    gate = lab2.get_task("quality_gate")
    best = {"model_name": "random_forest", "accuracy": accuracy}
    assert gate.python_callable(best, params={"min_accuracy": 0.72}) == expected


def test_summary_runs_after_either_branch(lab2):
    summary = lab2.get_task("pipeline_summary")
    assert summary.trigger_rule == "none_failed_min_one_success"
    assert summary.upstream_task_ids >= {"smoke_test_production_model", "reject_model"}


def test_run_params_defaults(lab2):
    assert lab2.params["min_accuracy"] == 0.72
    assert lab2.params["test_size"] == 0.3


def test_reliability_defaults(lab2):
    for task in lab2.tasks:
        assert task.execution_timeout is not None
        assert task.on_failure_callback
    assert lab2.get_task("load_data").retries == 2
    assert lab2.get_task("validate_data").retries == 0  # bad data fails fast


def test_asset_links_the_two_dags(dagbag, lab2):
    assert PRODUCTION_MODEL_ASSET in lab2.get_task("promote_model").outlets

    flask_dag = dagbag.dags["Airflow_Lab2_Flask"]
    assert type(flask_dag.timetable).__name__ == "AssetTriggeredTimetable"
    assert flask_dag.timetable.asset_condition.objects == (PRODUCTION_MODEL_ASSET,)
