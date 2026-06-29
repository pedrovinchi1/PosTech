import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException

from src.api.middleware import log_latency
from src.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)

MODEL_VERSION = "2.0.0"
_model = None
_preprocessor = None
_threshold = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _preprocessor, _threshold
    try:
        import pickle
        import torch
        from src.models.mlp import ChurnMLP

        with open("models/preprocessor_improved.pkl", "rb") as f:
            _preprocessor = pickle.load(f)

        meta = np.load("models/threshold_mlp_f1.npy")
        _threshold = float(meta)

        checkpoint = torch.load("models/best_mlp_improved.pt", map_location="cpu", weights_only=True)
        input_dim = checkpoint["network.0.weight"].shape[1]
        _model = ChurnMLP(input_dim=input_dim, hidden_dims=[256, 128, 64])
        _model.load_state_dict(checkpoint)
        _model.eval()
        logger.info("Model loaded", extra={"version": MODEL_VERSION, "threshold": _threshold})
    except FileNotFoundError:
        logger.warning("Model artifacts not found — /predict will return 503")
    yield


app = FastAPI(title="Churn Predictor API", version=MODEL_VERSION, lifespan=lifespan)
app.middleware("http")(log_latency)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd
    import torch

    from src.models.train_improved import engineer_features

    df = pd.DataFrame([payload.model_dump()])
    df = engineer_features(df)
    X = _preprocessor.transform(df)
    tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logit = _model(tensor)
        prob = float(torch.sigmoid(logit).item())

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prob >= _threshold,
        threshold_used=_threshold,
    )
