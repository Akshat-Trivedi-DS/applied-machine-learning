"""Automated tests for the FastAPI service.

Uses FastAPI's TestClient, which runs the app (including the lifespan
startup hook that loads the real trained model) in-process -- so these
tests require `models/best_model.joblib` to exist (run `python -m
src.train` first).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "limit_bal": 200000, "sex": 2, "education": 2, "marriage": 1, "age": 35,
    "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
    "bill_amt1": 50000, "bill_amt2": 48000, "bill_amt3": 46000,
    "bill_amt4": 44000, "bill_amt5": 42000, "bill_amt6": 40000,
    "pay_amt1": 3000, "pay_amt2": 3000, "pay_amt3": 3000,
    "pay_amt4": 3000, "pay_amt5": 3000, "pay_amt6": 3000,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_probability_and_risk_category(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_category"] in {"high_risk", "low_risk"}


def test_predict_rejects_invalid_age(client):
    bad_payload = {**VALID_PAYLOAD, "age": 5}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_education_code(client):
    bad_payload = {**VALID_PAYLOAD, "education": 9}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_negative_payment_amount(client):
    bad_payload = {**VALID_PAYLOAD, "pay_amt1": -100}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "limit_bal"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
