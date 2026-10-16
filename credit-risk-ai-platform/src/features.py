"""Feature engineering: credit utilization ratio and number of delayed payments.

Two new columns, each a straightforward calculation from columns already
in the dataset:

  - CREDIT_UTIL_RATIO = how much of the credit limit the customer is
    currently using (most recent bill / credit limit). A ratio close to
    or above 1 means they're maxed out, which is a classic red flag.

  - NUM_DELAYED_PAYMENTS = how many of the last 6 months had a late
    payment (PAY_1..PAY_6 > 0 means payment was delayed that many months).
"""

import numpy as np
import pandas as pd

PAY_STATUS_COLS = [f"PAY_{i}" for i in range(1, 7)]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["CREDIT_UTIL_RATIO"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["CREDIT_UTIL_RATIO"] = df["CREDIT_UTIL_RATIO"].fillna(0)

    df["NUM_DELAYED_PAYMENTS"] = (df[PAY_STATUS_COLS] > 0).sum(axis=1)

    return df


FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *PAY_STATUS_COLS,
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "CREDIT_UTIL_RATIO", "NUM_DELAYED_PAYMENTS",
]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]
