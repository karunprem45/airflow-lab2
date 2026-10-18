# File: main.py
"""
Airflow_Lab2 - model training pipeline with a model-selection quality gate.

    owner_task -> load_data -> validate_data -> preprocess
        -> train_candidate x N  (dynamic task mapping, runs in parallel)
        -> select_best
        -> quality_gate (branch) --> promote_model --> smoke_test_production_model
                                 \-> reject_model
        -> pipeline_summary

promote_model publishes PRODUCTION_MODEL_ASSET, which schedules the
Airflow_Lab2_Flask DAG (see Flask_API.py).
"""
from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

from src import model_development as md
from src.lab2_common import (
    PRODUCTION_MODEL_ASSET,
    dag_failure_alert,
    dag_success_alert,
    task_failure_alert,
    task_retry_alert,
)

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(seconds=15),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=10),
    "on_failure_callback": task_failure_alert,
    "on_retry_callback": task_retry_alert,
}


@dag(
    dag_id="Airflow_Lab2",
    description="Train several models in parallel, gate on accuracy, publish the best one",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["lab2", "ml"],
    owner_links={"Ramin Mohammadi": "https://github.com/raminmohammadi/MLOps/"},
    on_success_callback=dag_success_alert,
    on_failure_callback=dag_failure_alert,
    params={
        "min_accuracy": Param(
            0.95,
            type="number",
            minimum=0,
            maximum=1,
            description="Best model must reach this test accuracy to be promoted.",
        ),
        "test_size": Param(
            0.3,
            type="number",
            exclusiveMinimum=0,
            exclusiveMaximum=1,
            description="Fraction of rows held out for testing.",
        ),
        "models": Param(
            list(md.MODEL_FACTORIES),
            type="array",
            items={"type": "string", "enum": list(md.MODEL_FACTORIES)},
            minItems=1,
            description="Candidate models to train (one parallel task each).",
        ),
    },
    doc_md=__doc__,
)
def airflow_lab2():
    owner_task = BashOperator(
        task_id="task_using_linked_owner",
        bash_command="echo 1",
        owner="Ramin Mohammadi",
    )

    @task
    def load_data() -> str:
        return md.load_data()

    @task(retries=0)
    def validate_data(raw_path: str) -> str:
        # Bad data will not fix itself, so fail fast without retries.
        try:
            report = md.validate_data(raw_path)
        except ValueError as e:
            raise AirflowFailException(f"Data validation failed: {e}") from e
        print(f"Data validation passed: {report}")
        return raw_path

    @task
    def preprocess(raw_path: str, params: dict | None = None) -> str:
        return md.data_preprocessing(raw_path, test_size=params["test_size"])

    @task
    def list_candidates(params: dict | None = None) -> list[str]:
        return params["models"]

    @task(map_index_template="{{ model_name }}")
    def train_candidate(prep_path: str, model_name: str) -> dict:
        result = md.train_and_evaluate(prep_path, model_name)
        print(f"{model_name}: accuracy={result['accuracy']} f1={result['f1']}")
        return result

    @task
    def select_best(results: list[dict]) -> dict:
        best = md.select_best_model(list(results), get_current_context()["run_id"])
        print(f"Best candidate: {best}")
        return best

    @task.branch
    def quality_gate(best: dict, params: dict | None = None) -> str:
        threshold = params["min_accuracy"]
        if best["accuracy"] >= threshold:
            print(f"{best['model_name']} {best['accuracy']} >= {threshold}: promoting")
            return "promote_model"
        print(f"{best['model_name']} {best['accuracy']} < {threshold}: rejecting")
        return "reject_model"

    @task(outlets=[PRODUCTION_MODEL_ASSET])
    def promote_model(best: dict, params: dict | None = None) -> dict:
        context = get_current_context()
        metadata = md.promote_model(best, context["run_id"], params["min_accuracy"])
        # Attach the metrics to the Asset event so the consumer DAG can read them.
        context["outlet_events"][PRODUCTION_MODEL_ASSET].extra = metadata
        return metadata

    @task
    def smoke_test_production_model(prep_path: str, metadata: dict) -> float:
        score = md.score_production_model(prep_path)
        if abs(score - metadata["accuracy"]) > 1e-6:
            raise AirflowFailException(
                f"Reloaded model scored {score}, expected {metadata['accuracy']}"
            )
        print(f"Production model reloads correctly, score={score}")
        return score

    @task
    def reject_model(best: dict, params: dict | None = None) -> None:
        print(
            f"Not promoting {best['model_name']} (accuracy {best['accuracy']} "
            f"< {params['min_accuracy']}). The current production model is unchanged."
        )

    @task(trigger_rule="none_failed_min_one_success")
    def pipeline_summary(best: dict) -> None:
        context = get_current_context()
        promoted = context["ti"].xcom_pull(task_ids="promote_model")
        print(f"Run {context['run_id']} best={best['model_name']} promoted={bool(promoted)}")

    raw = load_data()
    validated = validate_data(raw)
    prep = preprocess(validated)
    results = train_candidate.partial(prep_path=prep).expand(model_name=list_candidates())
    best = select_best(results)
    gate = quality_gate(best)
    promoted = promote_model(best)
    smoke = smoke_test_production_model(prep, promoted)
    rejected = reject_model(best)
    summary = pipeline_summary(best)

    owner_task >> raw
    gate >> [promoted, rejected]
    [smoke, rejected] >> summary


airflow_lab2()
