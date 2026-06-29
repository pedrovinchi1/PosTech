import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "timestamp" in body


def test_predict_invalid_schema():
    r = client.post("/predict", json={"tenure": -1})
    assert r.status_code == 422


def test_predict_valid_or_503():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert isinstance(body["churn_prediction"], bool)
