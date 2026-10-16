"""Load the raw UCI dataset and clean undocumented category codes.

The UCI "default of credit card clients" dataset has a well-known data
quality issue: EDUCATION and MARRIAGE contain category codes that are not
documented in the data dictionary. We fold those into an explicit "other"
bucket instead of silently leaving them as unexplained integers.

EDUCATION dictionary says: 1=grad school, 2=university, 3=high school,
4=others. In the raw data you also see 0, 5, 6 -> undocumented, folded
into 4 (others).

MARRIAGE dictionary says: 1=married, 2=single, 3=other. In the raw data
you also see 0 -> undocumented, folded into 3 (other).
"""

import pandas as pd

RAW_PATH = "data/credit_card_default.csv"
TARGET_COL = "default payment next month"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={TARGET_COL: "default", "PAY_0": "PAY_1"})
    df = df.drop(columns=["ID"])
    return df


def clean_undocumented_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    return clean_undocumented_codes(load_raw(path))


if __name__ == "__main__":
    df = load_clean()
    print(df.shape)
    print("EDUCATION values after cleaning:", sorted(df["EDUCATION"].unique()))
    print("MARRIAGE values after cleaning:", sorted(df["MARRIAGE"].unique()))
    print("Default rate: %.2f%%" % (df["default"].mean() * 100))
