import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import build_preprocessor, prepare_xy


def _synthetic_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(18, 120, n),
        "TotalCharges": rng.uniform(0, 8000, n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "gender": rng.choice(["Male", "Female"], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n),
        "Churn": rng.choice(["Yes", "No"], n),
    })


def test_preprocessor_runs():
    df = _synthetic_df()
    X, y = prepare_xy(df)
    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(X)
    assert X_proc.shape[0] == len(df)
    assert X_proc.shape[1] > 0
    assert not np.isnan(X_proc).any()


def test_target_binary():
    df = _synthetic_df()
    _, y = prepare_xy(df)
    assert set(y.unique()).issubset({0, 1})
