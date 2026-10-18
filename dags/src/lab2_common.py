# File: src/lab2_common.py
"""
Pieces shared by both Lab 2 DAGs: the production-model Asset that links them,
and the success/failure callbacks.
"""
import logging
import os

from airflow.sdk import Asset

log = logging.getLogger(__name__)

# Data-aware scheduling: Airflow_Lab2 lists this Asset as an outlet of its
# promote task, and Airflow_Lab2_Flask is scheduled on it. A new production
# model therefore triggers the Flask DAG; a rejected model does not.
PRODUCTION_MODEL_ASSET = Asset(
    name="lab2_production_model",
    uri="file:///opt/airflow/model/production/model.sav",
    group="model",
)

# Optional e-mail alerts. Leave LAB2_ALERT_EMAIL empty to only log.
ALERT_EMAIL = os.getenv("LAB2_ALERT_EMAIL", "").strip()


def _email(context, subject: str, body: str) -> None:
    if not ALERT_EMAIL:
        return
    from airflow.providers.smtp.notifications.smtp import SmtpNotifier

    try:
        SmtpNotifier(to=ALERT_EMAIL, subject=subject, html_content=body).notify(context)
    except Exception:  # an alert problem must never fail the pipeline itself
        log.exception("Could not send alert e-mail to %s", ALERT_EMAIL)


def task_failure_alert(context) -> None:
    """Task-level on_failure_callback: runs once a task has used up its retries."""
    ti = context["ti"]
    error = context.get("exception")
    log.error("Task %s.%s failed (try %s): %s", ti.dag_id, ti.task_id, ti.try_number, error)
    _email(
        context,
        subject=f"[Airflow] Task failed: {ti.dag_id}.{ti.task_id}",
        body=f"<p>Run <b>{ti.run_id}</b>, task <b>{ti.task_id}</b> failed.</p><pre>{error}</pre>",
    )


def task_retry_alert(context) -> None:
    """Task-level on_retry_callback: logs every retry so flaky steps are visible."""
    ti = context["ti"]
    log.warning("Retrying %s.%s after error: %s", ti.dag_id, ti.task_id, context.get("exception"))


def dag_success_alert(context) -> None:
    """DAG-level on_success_callback (replaces the lab's old send_email task)."""
    dag_run = context["dag_run"]
    log.info("DAG run %s of %s succeeded", dag_run.run_id, dag_run.dag_id)
    _email(
        context,
        subject=f"[Airflow] {dag_run.dag_id} succeeded",
        body=f"<p>Run <b>{dag_run.run_id}</b> finished successfully.</p>",
    )


def dag_failure_alert(context) -> None:
    """DAG-level on_failure_callback."""
    dag_run = context["dag_run"]
    log.error("DAG run %s of %s failed", dag_run.run_id, dag_run.dag_id)
    _email(
        context,
        subject=f"[Airflow] {dag_run.dag_id} FAILED",
        body=f"<p>Run <b>{dag_run.run_id}</b> failed. Check the Airflow UI for logs.</p>",
    )
