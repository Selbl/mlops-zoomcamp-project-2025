# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Grade‑Class service  (Flask + MLflow server in one container)
# Build:  docker build -t gradeclass-service .
# Run :  docker run -p 9696:9696 -p 5000:5000 gradeclass-service
#        (5000 is optional but handy if you want to open the MLflow UI)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ---- system libs for xgboost -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN pip install --upgrade pip pipenv

WORKDIR /app

# ---- dependency layer --------------------------------------------------------
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy --ignore-pipfile

# ---- copy code (includes mlartifacts nested inside) --------------------------
COPY code/ code/

# ---- environment vars MLflow will use ----------------------------------------
ENV MLFLOW_ARTIFACT_ROOT=file:///app/code/mlartifacts         
ENV MLFLOW_BACKEND_URI=sqlite:////tmp/mlflow.db
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000
# ---- expose ports ------------------------------------------------------------
EXPOSE 9696 5000   

# ---- run both services in one shot ------------------------------------------
# • shell form `/bin/sh -c "cmd1 & cmd2"` lets us background MLflow (cmd1)
# • `exec` Gunicorn in the foreground so container stops when it stops
CMD /bin/sh -c "\
      mlflow server \
        --backend-store-uri ${MLFLOW_BACKEND_URI} \
        --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
        --artifacts-destination ${MLFLOW_ARTIFACT_ROOT} \
        --host 0.0.0.0 --port 5000 & \
      exec gunicorn --chdir code --bind 0.0.0.0:9696 flask_predict:app"