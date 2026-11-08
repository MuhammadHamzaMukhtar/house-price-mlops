# API tests: do the API's "doors" work?
# Needs champion.pkl (python export_model.py, or in CI: cp model.pkl champion.pkl).

import os

from fastapi.testclient import TestClient
from app import app, LOG_FILE

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_price():
    response = client.post("/predict", json={"rooms": 3, "size_sqft": 1350})
    assert response.status_code == 200
    assert response.json()["price_lakh"] > 0


def test_bad_input_is_rejected():
    response = client.post("/predict", json={"rooms": "three", "size_sqft": 1350})
    assert response.status_code == 422


def test_zero_rooms_is_rejected():
    response = client.post("/predict", json={"rooms": 0, "size_sqft": 1350})
    assert response.status_code == 422


def test_missing_field_is_rejected():
    response = client.post("/predict", json={"rooms": 3})
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text


def test_prediction_is_written_to_diary():
    client.post("/predict", json={"rooms": 4, "size_sqft": 1700})
    assert os.path.exists(LOG_FILE)
    with open(LOG_FILE) as f:
        lines = f.read().splitlines()
    assert lines[0] == "time,rooms,size_sqft,predicted_price_lakh"
    assert ",4,1700.0," in lines[-1]
