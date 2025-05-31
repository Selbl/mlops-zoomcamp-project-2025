#!/usr/bin/env python
# ------------------------------------------------------------
# XGBoost HPO → logs one MLflow run per grid point
# ------------------------------------------------------------
import os
import itertools
import click
import mlflow
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score

from prefect import flow,task
from sklearn.model_selection import train_test_split

import subprocess

TARGET_COL = "GradeClass"

@task(name='Connect-server')
def connect_prefect(url="http://127.0.0.1:4200/api"):
    # Config prefect
    # Define the command as a list of arguments
    command = [
        "prefect", "config", "set", f"PREFECT_API_URL={url}"
    ]
    # Run the command
    subprocess.run(command, capture_output=True, text=True)

@task(name="Read CSV")
def read_csv_as_xy(path: str):
    """Read a CSV, split into (X,y) and coerce booleans → ints."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL]).apply(
        lambda c: c.astype(int) if c.dtype == "bool" else c
    )
    y = df[TARGET_COL].astype(int)          # 1–4 ⇒ multi-class
    return X, y


@click.command()
@click.option("--data-path", default="../data/processed",
              help="Folder that contains train.csv, val.csv, test.csv")
@click.option("--experiment-name", default="gradeclass-xgb-hpo",
              help="MLflow experiment for HPO runs")

@flow(name="train-xgboost",log_prints=True)
def run_train(data_path: str, experiment_name: str):
    # --------------------------------------------------------
    # 0. prep
    # --------------------------------------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:5000")       # adapt if needed
    mlflow.set_experiment(experiment_name)
    mlflow.xgboost.autolog(disable=True)   # we'll log manually

    X_train, y_train = read_csv_as_xy(os.path.join(data_path, "train.csv"))
    X_val,   y_val   = read_csv_as_xy(os.path.join(data_path, "val.csv"))

    num_classes = len(np.unique(y_train))

    # --------------------------------------------------------
    # 1. hyper-parameter grid
    # (trim / extend at will – total combos = 3×3×3×2×2×2 = 216)
    # --------------------------------------------------------
    PARAM_GRID = {
        "n_estimators":      [200, 400, 600],
        "learning_rate":     [0.05, 0.10, 0.20],
        "max_depth":         [4, 6, 8],
        "min_child_weight":  [1, 5],
        "subsample":         [0.8, 1.0],
        "colsample_bytree":  [0.8, 1.0],
        "gamma":             [0, 0.1],
        # fixed:
        "objective":         ["multi:softprob"],
        "eval_metric":       ["mlogloss"],
        "random_state":      [42],
    }

    PARAM_GRID = {
        "n_estimators":      [200],
        "learning_rate":     [0.05],
        "max_depth":         [4, 6],
        "min_child_weight":  [1],
        "subsample":         [0.8],
        "colsample_bytree":  [0.8],
        "gamma":             [0, 0.1],
        # fixed:
        "objective":         ["multi:softprob"],
        "eval_metric":       ["mlogloss"],
        "random_state":      [42],
    }

    for idx, params in enumerate(ParameterGrid(PARAM_GRID), start=1):
        run_name = f"xgb-hpo-{idx:03d}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)

            model = xgb.XGBClassifier(
                **params,
                num_class=num_classes,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            acc   = accuracy_score(y_val, preds)
            f1    = f1_score(y_val, preds, average="macro")

            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_f1_macro", f1)
            mlflow.xgboost.log_model(model, artifact_path="model")

            print(f"[{run_name}] acc={acc:0.4f}  f1={f1:0.4f}")

    print("\n🎉  Hyper-parameter search finished – inspect your MLflow UI!")

if __name__ == "__main__":
    connect_prefect()
    run_train()
