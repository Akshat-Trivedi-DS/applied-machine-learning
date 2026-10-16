"""Loads the trained model once at startup and serves predictions.

Kept as a plain class (not re-loaded per request) so the joblib file is
only read once, not on every API call.
"""

import json
import os

import joblib
import pandas as pd

from api.schemas import CreditApplication
from src.features import FEATURE_COLUMNS, add_engineered_features

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
RISK_THRESHOLD = 0.5


class ModelService:
    def __init__(self):
        self.model = joblib.load(f"{MODEL_DIR}/best_model.joblib")
        with open(f"{MODEL_DIR}/metrics.json") as f:
            self.model_name = json.load(f)["best_model"]

    def _to_raw_dataframe(self, application: CreditApplication) -> pd.DataFrame:
        row = {
            "LIMIT_BAL": application.limit_bal,
            "SEX": application.sex,
            "EDUCATION": application.education,
            "MARRIAGE": application.marriage,
            "AGE": application.age,
            "PAY_1": application.pay_1, "PAY_2": application.pay_2, "PAY_3": application.pay_3,
            "PAY_4": application.pay_4, "PAY_5": application.pay_5, "PAY_6": application.pay_6,
            "BILL_AMT1": application.bill_amt1, "BILL_AMT2": application.bill_amt2,
            "BILL_AMT3": application.bill_amt3, "BILL_AMT4": application.bill_amt4,
            "BILL_AMT5": application.bill_amt5, "BILL_AMT6": application.bill_amt6,
            "PAY_AMT1": application.pay_amt1, "PAY_AMT2": application.pay_amt2,
            "PAY_AMT3": application.pay_amt3, "PAY_AMT4": application.pay_amt4,
            "PAY_AMT5": application.pay_amt5, "PAY_AMT6": application.pay_amt6,
        }
        return pd.DataFrame([row])

    def predict(self, application: CreditApplication) -> dict:
        raw_df = self._to_raw_dataframe(application)
        features_df = add_engineered_features(raw_df)[FEATURE_COLUMNS]

        probability = float(self.model.predict_proba(features_df)[0, 1])
        risk_category = "high_risk" if probability >= RISK_THRESHOLD else "low_risk"

        return {
            "default_probability": round(probability, 4),
            "risk_category": risk_category,
        }
