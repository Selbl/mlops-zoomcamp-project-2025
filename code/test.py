# optional: save processed features
import os

# import ast
import argparse
import mlflow
import pandas as pd

# import numpy as np
# import xgboost as xgb
# from distutils.util import strtobool

from mlflow.tracking import MlflowClient

# from mlflow.models.signature import ModelSignature

# add this at the very top with the other imports
from mlflow.types.schema import DataType


def strtobool(s):
    if s in ["True", "true", "TRUE"]:
        return True
    elif s in ["False", "false", "FALSE"]:
        return False
    raise ValueError("Invalid string boolean value")


# ------------------------------------------------------------------
def align_to_signature(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Pad / prune columns AND cast them to the dtypes specified in the
    MLflow model signature. Works even for single-row inputs.
    """
    sig = model.metadata.get_input_schema()
    cols = [spec.name for spec in sig.inputs]

    # 1️⃣  add missing columns (0) / drop extras, keep order
    aligned = df.reindex(columns=cols, fill_value=0)

    # 2️⃣  cast to required dtypes
    for spec in sig.inputs:
        col = spec.name
        dtype = spec.type  # this is a DataType enum
        if dtype == DataType.long:
            # long ⇒ 64-bit signed int
            aligned[col] = aligned[col].astype("int64", copy=False)
        elif dtype == DataType.double:
            aligned[col] = aligned[col].astype("float64", copy=False)
        # bools or objects will be coerced too
    return aligned


# ------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GradeClass inference on a single observation."
    )
    parser.add_argument(
        "--data_location",
        type=str,
        default="../data/sample_test_input.csv",
        help="CSV file with exactly one row of raw features.",
    )
    parser.add_argument(
        "--save_processed",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="If True, save the one-hot encoded row next to the CSV.",
    )
    return parser.parse_args()


# Define helper one hot encoding function
def one_hot_encoding(df):

    # Define categoric columns
    cat_cols = [
        "Age",
        "Gender",
        "Ethnicity",
        "ParentalEducation",
        "Tutoring",
        "ParentalSupport",
        "Extracurricular",
        "Sports",
        "Music",
        "Volunteering",
    ]

    # Create mapping
    gender_map = {0: "Male", 1: "Female"}
    ethnicity_map = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}
    parental_education_map = {
        0: "None",
        1: "High School",
        2: "Some College",
        3: "Bachelor's",
        4: "Higher",
    }
    tutoring_map = {0: "No", 1: "Yes"}
    parental_support_map = {
        0: "None",
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High",
    }
    yes_no_map = {0: "No", 1: "Yes"}

    # Apply the mappings to the DataFrame columns
    df["Gender"] = df["Gender"].map(gender_map)
    df["Ethnicity"] = df["Ethnicity"].map(ethnicity_map)
    df["ParentalEducation"] = df["ParentalEducation"].map(parental_education_map)
    df["Tutoring"] = df["Tutoring"].map(tutoring_map)
    df["ParentalSupport"] = df["ParentalSupport"].map(parental_support_map)
    df["Extracurricular"] = df["Extracurricular"].map(yes_no_map)
    df["Sports"] = df["Sports"].map(yes_no_map)
    df["Music"] = df["Music"].map(yes_no_map)
    df["Volunteering"] = df["Volunteering"].map(yes_no_map)

    # One-hot encode
    df_with_dummies = pd.get_dummies(
        df, columns=cat_cols, prefix=cat_cols, drop_first=False
    )

    # Drop student ID
    df_with_dummies.drop("StudentID", axis=1, inplace=True)

    return df_with_dummies


def get_model_uri(model_name: str, stage: str) -> str:
    client = MlflowClient()
    # try preferred stage first
    latest = client.get_latest_versions(model_name, stages=[stage])
    if not latest:
        # fall back to “Production”, then to the very latest version
        for alt in ("Production", "None"):
            latest = client.get_latest_versions(model_name, stages=[alt])
            if latest:
                break
    model_version = latest[0].version
    print(
        f"📦  Using model '{model_name}' version {model_version} "
        f"(stage={latest[0].current_stage})"
    )
    return f"models:/{model_name}/{model_version}"


def main():
    # Define grade class map for printing
    grade_class_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
    # Parse arguments
    args = parse_args()
    # Load data
    df = pd.read_csv(args.data_location, index_col=False)
    # Pre-process
    df = one_hot_encoding(df)
    # Connect to MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # adjust if necessary
    # Retrieve model
    model_uri = get_model_uri("gradeclass-xgb-classifier", "Staging")
    model = mlflow.pyfunc.load_model(model_uri)
    df_aligned = align_to_signature(df, model)

    # Return predicted class
    pred_class = model.predict(df_aligned)[0]

    # Print
    print("\n🔮  Prediction")
    print("--------------")
    print(f"GradeClass  : {grade_class_map[pred_class]}")

    # optional: save processed features
    if args.save_processed:
        out_path = os.path.splitext(args.data_location)[0] + "_encoded.csv"
        df_aligned.to_csv(out_path, index=False)
        print(f"\n💾  Saved encoded row to: {out_path}")


if __name__ == "__main__":
    main()
