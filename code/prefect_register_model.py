#!/usr/bin/env python
# ------------------------------------------------------------
# Select best HPO run by val_f1_macro → retrain on (train+val)
# evaluate on test  → register in MLflow Model Registry
# ------------------------------------------------------------
import os
import ast
import click
import mlflow
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import accuracy_score, f1_score, classification_report
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from prefect import flow,task
from sklearn.model_selection import train_test_split

import subprocess

TARGET_COL        = "GradeClass"
HPO_EXPERIMENT    = "gradeclass-xgb-hpo"      # must match the HPO script
FINAL_EXPERIMENT  = "gradeclass-xgb-final"
MODEL_NAME        = "gradeclass-xgb-classifier"

@task(name='Connect-server')
def connect_prefect(url="http://127.0.0.1:4200/api"):
    # Config prefect
    # Define the command as a list of arguments
    command = [
        "prefect", "config", "set", f"PREFECT_API_URL={url}"
    ]
    # Run the command
    subprocess.run(command, capture_output=True, text=True)

@task(name="Read_csv")
def read_csv_as_xy(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL]).apply(
        lambda c: c.astype(int) if c.dtype == "bool" else c
    )
    y = df[TARGET_COL].astype(int)
    return X, y

@task(name="cast-parameters")
def cast_params(params: dict):
    """Cast MLflow-stringified params back to proper dtypes."""
    out = {}
    for k, v in params.items():
        if v == "None":
            out[k] = None
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = ast.literal_eval(v)
            continue
        except Exception:
            pass
        out[k] = v
    return out


@click.command()
@click.option("--data-path", default="../data/processed",
              help="Folder with train.csv, val.csv, test.csv")
@click.option("--model_name", default=MODEL_NAME,
              help="Name inside MLflow Model Registry")

@flow(name='register-model',log_prints=True)
def main(data_path: str, model_name: str):
    # --------------------------------------------------------
    # 1. locate best HPO run (highest val_f1_macro)
    # --------------------------------------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:5000")    # adjust if needed
    client = MlflowClient()

    exp_id = client.get_experiment_by_name(HPO_EXPERIMENT).experiment_id
    best_run = client.search_runs(
        experiment_ids=[exp_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.val_f1_macro DESC"]
    )[0]

    best_params = cast_params(best_run.data.params)
    best_f1     = best_run.data.metrics["val_f1_macro"]
    print(f"🔎  Best HPO run {best_run.info.run_id}  "
          f"val_macro_F1={best_f1:.4f}")

    # --------------------------------------------------------
    # 2. load data & retrain on (train + val)
    # --------------------------------------------------------
    X_train, y_train = read_csv_as_xy(os.path.join(data_path, "train.csv"))
    X_val,   y_val   = read_csv_as_xy(os.path.join(data_path, "val.csv"))
    X_test,  y_test  = read_csv_as_xy(os.path.join(data_path, "test.csv"))

    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = pd.concat([y_train, y_val], axis=0)
    num_classes = len(np.unique(y_trval))

    best_params.update({
        "objective":   "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class":   num_classes,
        "n_jobs":      -1,
        "random_state": 42,
    })

    mlflow.set_experiment(FINAL_EXPERIMENT)
    mlflow.xgboost.autolog()

    with mlflow.start_run(run_name="xgb-final-f1") as run:
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_trval, y_trval)

        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro")

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_macro", f1)
        mlflow.log_dict(
            classification_report(y_test, preds, output_dict=True),
            "classification_report.json"
        )

        # register
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        print(f"✅  Logged run: {run.info.run_id}")
        print(f"📦  Registered model: {model_name} v{mv.version}")
        print(f"🎯  Test  acc: {acc:0.4f}  macro-F1: {f1:0.4f}")


if __name__ == "__main__":
    connect_prefect()
    main()
