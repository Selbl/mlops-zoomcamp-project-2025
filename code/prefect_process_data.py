import argparse

# from distutils.util import strtobool
import pandas as pd

# import numpy as np
# import matplotlib.pyplot as plt

from prefect import flow, task
from sklearn.model_selection import train_test_split

import subprocess


def strtobool(s):
    if s in ["True", "true", "TRUE"]:
        return True
    elif s in ["False", "false", "FALSE"]:
        return False
    raise ValueError("Invalid string boolean value")


@task(name="Parser")
def parse_args():
    parser = argparse.ArgumentParser(
        description="Split student performance data and optionally save raw splits."
    )

    parser.add_argument(
        "--data_location",
        dest="data_location",
        type=str,
        default="../data/students_performance.csv",
        help="Location to the dataset to process. Has to be .csv",
    )

    parser.add_argument(
        "--save-raw",
        dest="save_raw",
        type=lambda x: strtobool(x),
        default=False,
        help="Whether to save raw train/val/test CSVs (True or False). Default is True.",
    )
    return parser.parse_args()


@task(name="Sample split")
def train_val_test_split(df, save_raw=True):
    # Format covariates and target variable
    X = df.drop(columns=["GradeClass"])
    y = df["GradeClass"]

    # Apply first split
    X_temp, X_val, y_temp, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Split again
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )

    # Extract dataframes
    df_train = df[df["StudentID"].isin(X_train["StudentID"])]
    df_val = df[df["StudentID"].isin(X_val["StudentID"])]
    df_test = df[df["StudentID"].isin(X_test["StudentID"])]

    # Save to file
    if save_raw:
        df_train.to_csv("../data/raw/train.csv", index=False)
        df_val.to_csv("../data/raw/val.csv", index=False)
        df_test.to_csv("../data/raw/test.csv", index=False)

    return df_train, df_val, df_test


@task(name="One hot encoding")
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


@task(name="Connect-server")
def connect_prefect(url="http://127.0.0.1:4200/api"):
    # Config prefect
    # Define the command as a list of arguments
    command = ["prefect", "config", "set", f"PREFECT_API_URL={url}"]
    # Run the command
    subprocess.run(command, capture_output=True, text=True)


@flow(name="process-data", log_prints=True)
def main():
    # Parse arguments
    args = parse_args()
    # Connect to prefect
    connect_prefect()
    # Load dataframe
    df = pd.read_csv(args.data_location, index_col=False)
    # Drop redundant target variable
    df.drop("GPA", axis=1, inplace=True)
    # Perform test,train,split
    df_train, df_val, df_test = train_val_test_split(df, save_raw=args.save_raw)
    # One-hot encoding
    df_train, df_val, df_test = (
        one_hot_encoding(df_train),
        one_hot_encoding(df_val),
        one_hot_encoding(df_test),
    )
    # Save
    df_train.to_csv("../data/processed/train.csv", index=False)
    df_val.to_csv("../data/processed/val.csv", index=False)
    df_test.to_csv("../data/processed/test.csv", index=False)


if __name__ == "__main__":
    # Main function
    main()
