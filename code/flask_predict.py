# gradeclass_service.py
# ---------------------------------------------------------------------------
# REST service that re‑uses the *prefect_test.py* pipeline to score a single
# student record.  Launch with:
#
#     python gradeclass_service.py
#
# Then POST a JSON object with the raw, *numeric‑coded* fields described in
# prefect_test.py to  http://localhost:9696/predict
# ---------------------------------------------------------------------------

import mlflow
import pandas as pd
from flask import Flask, request, jsonify

# ⬇️  re‑use the helper functions already written in prefect_test.py
from prefect_test import (
    one_hot_encoding,        # Prefect task ⇒ callable here
    align_to_signature,      # utility to fit MLflow signature
    get_model_uri            # resolves latest model version
)

# ────────────────────────────────────────────────────────────────────────────
# Configuration – edit if your MLflow host or model name differ
# ────────────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "gradeclass-xgb-classifier"
MODEL_STAGE         = "Staging"            # fallback to Production/None handled
GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}

# connect to MLflow once, load the model once
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = get_model_uri(MODEL_NAME, MODEL_STAGE)
model     = mlflow.pyfunc.load_model(model_uri)

# ────────────────────────────────────────────────────────────────────────────
# Flask app
# ────────────────────────────────────────────────────────────────────────────
app = Flask("gradeclass-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Expects JSON with the raw integer‑coded student fields.
    Returns the predicted grade class (letter + numeric index).
    """
    record = request.get_json()

    # 1️⃣  convert to DataFrame (single row)
    df = pd.DataFrame([record])

    # 2️⃣  apply same pre‑processing used in training
    df = one_hot_encoding(df)

    # 3️⃣  align to the model’s input signature
    df_aligned = align_to_signature(df, model)

    # 4️⃣  predict
    pred_class = int(model.predict(df_aligned)[0])

    return jsonify(
        {
            "grade_class_index": pred_class,
            "grade_class": GRADE_MAP.get(pred_class, "Unknown"),
        }
    )


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # run on all interfaces so `docker run -p 9696:9696 …` also works
    app.run(host="0.0.0.0", port=9696, debug=False)
