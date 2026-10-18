"""
Airflow_Lab2_Flask - serves a status page for the Airflow_Lab2 pipeline.

Scheduled on PRODUCTION_MODEL_ASSET (data-aware scheduling) instead of being
pushed by a TriggerDagRunOperator: it runs only when Airflow_Lab2 actually
publishes a new production model.

    log_new_model -> serve_flask_api

serve_flask_api starts the Flask server on port 5555 the first time. On later
runs the server is already up (and re-reads the model files on every request),
so the task skips itself instead of failing with "port already in use".
"""
from __future__ import annotations

import json
import logging
import os
import time

import pendulum
import requests
from airflow.sdk import dag, get_current_context, task
from airflow.sdk.exceptions import AirflowSkipException
from flask import Flask, jsonify, redirect, render_template

from src.lab2_common import PRODUCTION_MODEL_ASSET, task_failure_alert
from src.model_development import LEADERBOARD_PATH, PRODUCTION_METADATA

logger = logging.getLogger(__name__)

# ---------- Config ----------
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-apiserver:8080")
AF_USER = os.getenv("AIRFLOW_USERNAME", "airflow")
AF_PASS = os.getenv("AIRFLOW_PASSWORD", "airflow")
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "Airflow_Lab2")
FLASK_PORT = int(os.getenv("LAB2_FLASK_PORT", "5555"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Cached token to avoid requesting a new token on every HTTP request
_CACHED_TOKEN = None
_TOKEN_EXPIRY = 0


def get_auth_token():
    global _CACHED_TOKEN, _TOKEN_EXPIRY
    if _CACHED_TOKEN and time.time() < _TOKEN_EXPIRY - 60:
        return _CACHED_TOKEN

    resp = requests.post(
        f"{WEBSERVER}/auth/token",
        json={"username": AF_USER, "password": AF_PASS},
        timeout=10,
    )
    if resp.status_code in (200, 201):
        _CACHED_TOKEN = resp.json().get("access_token")
        _TOKEN_EXPIRY = time.time() + 3600
        return _CACHED_TOKEN

    logger.error("Failed to authenticate with Airflow: %s %s", resp.status_code, resp.text)
    return None


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def get_latest_run_info():
    """Latest Airflow_Lab2 run from the REST API, plus the model files on disk."""
    info = {
        "state": "unknown",
        "run_id": "N/A",
        "logical_date": "N/A",
        "start_date": "N/A",
        "end_date": "N/A",
        "note": "",
        "production": _read_json(PRODUCTION_METADATA),
        "leaderboard": _read_json(LEADERBOARD_PATH) or {"run_id": None, "models": []},
    }
    # Did the latest training run promote its model or reject it at the gate?
    prod = info["production"]
    info["latest_promoted"] = bool(
        prod and prod.get("promoted_by_run") == info["leaderboard"]["run_id"]
    )

    token = get_auth_token()
    if not token:
        info["note"] = "Authentication failed (could not get token)"
        return False, info

    url = f"{WEBSERVER}/api/v2/dags/{TARGET_DAG_ID}/dagRuns?order_by=-run_after&limit=1"
    try:
        r = requests.get(
            url,
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code != 200:
            info["note"] = f"API Error {r.status_code}"
            return False, info

        runs = r.json().get("dag_runs", [])
        if not runs:
            info["note"] = "Waiting for DAG run to appear..."
            return False, info

        run = runs[0]
        state = run.get("state", "unknown")
        info.update({
            "state": state,
            "run_id": run.get("dag_run_id", "N/A"),
            "logical_date": run.get("logical_date") or run.get("run_after", "N/A"),
            "start_date": run.get("start_date", "N/A"),
            "end_date": run.get("end_date", "N/A"),
        })
        if state in ("queued", "running"):
            info["note"] = "Model pipeline is still in progress..."
            return False, info
        return state == "success", info

    except Exception as e:
        info["note"] = f"Network issue: {e}"
        return False, info


@app.route("/")
def index():
    ok, _ = get_latest_run_info()
    return redirect("/success" if ok else "/failure")


@app.route("/success")
def success():
    _, info = get_latest_run_info()
    return render_template("success.html", **info)


@app.route("/failure")
def failure():
    _, info = get_latest_run_info()
    return render_template("failure.html", **info)


@app.route("/api/status")
def api_status():
    ok, info = get_latest_run_info()
    return jsonify({"pipeline_ok": ok, **info})


@app.route("/health")
def health():
    return "ok", 200


def flask_is_running() -> bool:
    try:
        return requests.get(f"http://localhost:{FLASK_PORT}/health", timeout=2).ok
    except requests.RequestException:
        return False


# ---------- DAG Definition ----------
@dag(
    dag_id="Airflow_Lab2_Flask",
    description="Status page for Airflow_Lab2, refreshed whenever a new model is published",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=[PRODUCTION_MODEL_ASSET],
    catchup=False,
    is_paused_upon_creation=False,
    default_args={"retries": 0, "on_failure_callback": task_failure_alert},
    tags=["lab2", "serving"],
    doc_md=__doc__,
)
def airflow_lab2_flask():
    @task
    def log_new_model() -> dict | None:
        """Print the metrics that Airflow_Lab2 attached to the Asset event."""
        events = get_current_context()["triggering_asset_events"][PRODUCTION_MODEL_ASSET]
        if not events:
            print("Run was not triggered by an Asset event (manual run).")
            return None
        extra = dict(events[-1].extra)
        print(f"New production model published: {extra}")
        return extra

    @task
    def serve_flask_api() -> None:
        if flask_is_running():
            raise AirflowSkipException(
                f"Flask is already serving on :{FLASK_PORT}; it reads the new model on the next request."
            )
        print(f"Starting Flask on 0.0.0.0:{FLASK_PORT}, checking {WEBSERVER}...", flush=True)
        # Blocks this worker slot for as long as the server runs.
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

    log_new_model() >> serve_flask_api()


airflow_lab2_flask()
