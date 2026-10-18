#!/bin/bash
# One-time setup for Airflow Lab 2. Run from this folder: ./setup.sh
set -e
cd "$(dirname "$0")"

mkdir -p ./logs ./plugins ./config ./working_data ./model

if [ ! -f .env ]; then
  cp .env.example .env
fi
# Airflow containers run as your user so they can write to the mounted folders.
if grep -q '^AIRFLOW_UID=' .env; then
  sed -i.bak "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env && rm -f .env.bak
else
  echo "AIRFLOW_UID=$(id -u)" >> .env
fi

# Create the metadata database and the airflow/airflow login.
docker compose up airflow-init
echo "Setup done. Start Airflow with: docker compose up -d"
