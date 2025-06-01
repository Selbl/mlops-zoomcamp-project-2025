import pytest
import pandas as pd
from code.prefect_process_data import strtobool
from code.process_data import one_hot_encoding


def test_bool():
    assert strtobool("True")
    assert not strtobool("False")
    with pytest.raises(ValueError, match="Invalid string boolean value"):
        strtobool("foobar")


def test_one_hot_encoding():

    student = {
        "Age": 16,
        "Gender": 1,
        "Ethnicity": 2,
        "ParentalEducation": 3,
        "Tutoring": 0,
        "ParentalSupport": 2,
        "Extracurricular": 1,
        "Sports": 0,
        "Music": 1,
        "Volunteering": 0,
        "StudentID": 123,
    }

    feats = one_hot_encoding(pd.DataFrame(student, index=[0]))
    # expected keys present
    assert "Age_16" in feats.columns
    assert "Gender_Female" in feats.columns
    assert "Volunteering_No" in feats.columns
    assert "StudentID" not in feats.columns
