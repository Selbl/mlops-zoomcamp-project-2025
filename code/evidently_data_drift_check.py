import pandas as pd

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset

from evidently.tests.numerical_tests import TestStatus as num_test_status
from evidently.tests.categorical_tests import TestStatus as cat_test_status

from prefect import flow, task


# Define global failure variables for easy access
num_fail = num_test_status.FAIL
cat_fail = cat_test_status.FAIL


@task(name="load_csv")
def load_csv():
    # Use raw versions to leverage evidently's API
    train = pd.read_csv("../data/raw/train.csv", index_col=False)
    val = pd.read_csv("../data/raw/val.csv", index_col=False)

    return train, val


@task(name="prepare_report")
def drift_preparation(train, val):
    # Define schema
    schema = DataDefinition(
        numerical_columns=["StudyTimeWeekly", "Absences"],
        categorical_columns=[
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
            "GradeClass",
        ],
    )

    # Format as Evidently dataset objects
    train_ev = Dataset.from_pandas(pd.DataFrame(train), data_definition=schema)

    val_ev = Dataset.from_pandas(pd.DataFrame(val), data_definition=schema)

    report = Report([DataDriftPreset()], include_tests="True")

    my_eval = report.run(train_ev, val_ev)

    return my_eval


@task(name="check_drift")
def check_drift(my_eval):
    for i in range(14):
        value = my_eval.dict()["tests"][i]["status"]
        if value in [num_fail, cat_fail]:
            # Check for the GradeClass column
            if i == 13:
                print("OH NO!\n")
                print("It appears you have value drift for the dependent variable\n")
                print("Here is the log:\n")
                print(my_eval.dict()["tests"][i]["description"])
                print(" ")
                print("You should strongly consider re-generating the data split\n")
                print("If not, it is possible that your model will be ineffective\n")

                continue

            # Check for the others
            print("Uh oh!")
            print("There seems to be a drift in a column!\n")
            print(my_eval.dict()["tests"][i]["name"])
            print(" ")
            print(my_eval.dict()["tests"][i]["description"])
            print(" ")
            print("You might want to consider re-generating the data split\n")


@flow(name="data-drift-main-flow")
def main():
    print("Loading CSV...\n")
    train, val = load_csv()
    print("Performing data drift data preparation...\n")
    my_eval = drift_preparation(train, val)
    print("Checking for potential drifts...\n")
    check_drift(my_eval)


if __name__ == "__main__":
    main()
