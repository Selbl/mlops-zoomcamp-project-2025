# prefect_orchestrate_pipeline.py
# -------------------------------------------------------------------------
# Run the whole Prefect pipeline end‑to‑end.
#
#   python prefect_orchestrate_pipeline.py --process True   # incl. data prep
#   python prefect_orchestrate_pipeline.py                  # skip data prep
# -------------------------------------------------------------------------

import argparse
import sys
from contextlib import contextmanager
#rom distutils.util import strtobool

from prefect import flow, get_run_logger

# ── import the Click commands from the three scripts ──────────────────────
from prefect_process_data import main as _process_cmd
from prefect_train_xgb import run_train as _train_cmd
from prefect_register_model import main as _register_cmd

from pathlib import Path

# Helper
def strtobool(s):
    if s in ['True','true','TRUE']:
        return True
    elif s in ['False','false','FALSE']:
        return False
    raise ValueError("Invalid string boolean value")

# ───────────── helper to unwrap Click → plain function ────────────────────
def unwrap(cmd_or_func):
    """Return .callback if *cmd_or_func* is a Click command; else itself."""
    return getattr(cmd_or_func, "callback", cmd_or_func)


# ───────────── helper to patch sys.argv temporarily ───────────────────────
@contextmanager
def argv_for(*args: str):
    old = sys.argv[:]
    sys.argv = [old[0], *args]
    try:
        yield
    finally:
        sys.argv = old


# ───────────── unwrap the three commands ──────────────────────────────────
process_data_flow   = unwrap(_process_cmd)    # expects its own argparse
train_flow          = unwrap(_train_cmd)      # signature: (data_path, experiment_name)
register_model_flow = unwrap(_register_cmd)   # signature: (data_path, model_name)


# ───────────── configuration ──────────────────────────────────────────────
HERE      = Path(__file__).resolve()           # …/code/prefect_orchestrate_pipeline.py
PROJECT   = HERE.parent.parent                 # …/mlops-zoomcamp-project-2025

DATA_DIR  = PROJECT / "data"
RAW_DATA_CSV  = DATA_DIR / "students_performance.csv"
PROCESSED_DIR = DATA_DIR / "processed"

#RAW_DATA_CSV    = "../data/students_performance.csv"   # raw file for step 1
#PROCESSED_DIR   = "../data/processed"                  # folder of train/val/test
EXPERIMENT_NAME = "gradeclass-xgb-hpo-3"
MODEL_NAME      = "gradeclass-xgb-3"


# ───────────── master Prefect flow ────────────────────────────────────────
@flow(name="model-deployment-orchestrator", log_prints=True)
def orchestrate(process: bool = False) -> None:
    log = get_run_logger()

    # 1️⃣ Data processing ---------------------------------------------------
    if process:
        log.info("Step 1/3 ▸ running data‑processing flow …")
        # The inner script still uses its own argparse → patch argv
        #with argv_for("--data_location", RAW_DATA_CSV, "--save-raw", "False"):
        with argv_for("--data_location", str(RAW_DATA_CSV), "--save-raw", "False"):
            process_data_flow()
    else:
        log.info("Step 1/3 ▸ skipping data‑processing flow (process=False)")

    # 2️⃣ Training / hyper‑parameter search --------------------------------
    log.info("Step 2/3 ▸ hyper‑parameter search with XGBoost …")
    #train_flow(data_path=PROCESSED_DIR, experiment_name=EXPERIMENT_NAME)
    train_flow(data_path=str(PROCESSED_DIR), experiment_name=EXPERIMENT_NAME)

    # 3️⃣ Re‑train best + register model -----------------------------------
    log.info("Step 3/3 ▸ registering best model …")
    #register_model_flow(data_path=PROCESSED_DIR, model_name=MODEL_NAME)
    register_model_flow(data_path=str(PROCESSED_DIR), model_name=MODEL_NAME)

    log.info("🎉  Pipeline finished successfully")


# ───────────── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Prefect pipeline.")
    parser.add_argument(
        "--process",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Run data‑processing step first (default: False)."
    )
    args = parser.parse_args()
    orchestrate(process=args.process)
