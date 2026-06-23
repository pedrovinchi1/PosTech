from fastapi.testclient import TestClient

from src.api.main import app


def valid_payload() -> dict:
    return {
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
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True
    assert "threshold" in response.json()
    assert "timestamp" in response.json()


def test_predict_valid():
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["churn_prediction"], bool)
    assert 0 <= body["threshold_used"] <= 1


def test_predict_invalid_schema():
    payload = valid_payload()
    payload["tenure"] = -1

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422