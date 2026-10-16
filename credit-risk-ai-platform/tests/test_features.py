"""Unit tests for data cleaning and feature engineering logic."""

import pandas as pd

from src.data_prep import clean_undocumented_codes
from src.features import add_engineered_features


def test_clean_undocumented_education_codes():
    df = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6], "MARRIAGE": [1, 1, 1, 1, 1, 1, 1]})
    cleaned = clean_undocumented_codes(df)
    assert set(cleaned["EDUCATION"].unique()) <= {1, 2, 3, 4}


def test_clean_undocumented_marriage_codes():
    df = pd.DataFrame({"EDUCATION": [1, 1, 1], "MARRIAGE": [0, 1, 2]})
    cleaned = clean_undocumented_codes(df)
    assert set(cleaned["MARRIAGE"].unique()) <= {1, 2, 3}


def test_credit_util_ratio_is_bill_over_limit():
    df = pd.DataFrame({
        "LIMIT_BAL": [100000],
        "BILL_AMT1": [50000],
        **{f"PAY_{i}": [0] for i in range(1, 7)},
    })
    result = add_engineered_features(df)
    assert result.loc[0, "CREDIT_UTIL_RATIO"] == 0.5


def test_credit_util_ratio_handles_zero_credit_limit():
    df = pd.DataFrame({
        "LIMIT_BAL": [0],
        "BILL_AMT1": [1000],
        **{f"PAY_{i}": [0] for i in range(1, 7)},
    })
    result = add_engineered_features(df)
    assert result.loc[0, "CREDIT_UTIL_RATIO"] == 0.0


def test_num_delayed_payments_counts_positive_pay_status():
    df = pd.DataFrame({
        "LIMIT_BAL": [100000],
        "BILL_AMT1": [1000],
        "PAY_1": [2], "PAY_2": [1], "PAY_3": [-1], "PAY_4": [0], "PAY_5": [3], "PAY_6": [-2],
    })
    result = add_engineered_features(df)
    assert result.loc[0, "NUM_DELAYED_PAYMENTS"] == 3
