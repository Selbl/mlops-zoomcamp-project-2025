# ------------------------------------------------------------
# Base image
# ------------------------------------------------------------
FROM python:3.12-slim

# ------------------------------------------------------------
# System setup: update apt and install any build deps you need
# (none required for this project, but leaving pattern)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python deps: pip↑, pipenv, then project + dev tools
# ------------------------------------------------------------
RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

# Copy Pipenv files first so Docker can cache layer
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

# ------------------------------------------------------------
# Copy project source
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# Environment variables for MLflow
# (the actual paths are overridden at `docker run` via -e or -v)
# ------------------------------------------------------------
ENV MLFLOW_BACKEND_URI=sqlite:////app/code/mlflow.db
ENV MLFLOW_ARTIFACT_ROOT=file:///app/code/mlartifacts

# ------------------------------------------------------------
# Ports: 9696 (Flask API), 5000 (MLflow UI), 4200 (Prefect UI)
# ------------------------------------------------------------
EXPOSE 9696
EXPOSE 5000
EXPOSE 4200

# ------------------------------------------------------------
# Entrypoint:
#   1. Prefect server (port 4200)
#   2. MLflow tracking server (port 5000)
#   3. Gunicorn Flask API (port 9696)  ← keeps container foreground
# ------------------------------------------------------------
CMD /bin/sh -c "\
  prefect server start --host 0.0.0.0 & \
  mlflow server \
      --backend-store-uri ${MLFLOW_BACKEND_URI} \
      --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
      --artifacts-destination ${MLFLOW_ARTIFACT_ROOT} \
      --host 0.0.0.0 --port 5000 & \
  exec gunicorn --chdir code --bind 0.0.0.0:9696 flask_predict:app \
"
