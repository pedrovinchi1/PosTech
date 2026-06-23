import logging
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.predictor import ChurnPredictor
from src.api.schemas import CustomerFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Predictor API",
    version="1.0.0",
    description="API para previsão de churn de clientes Telco.",
)

predictor: ChurnPredictor | None = None


@app.on_event("startup")
def startup_event():
    global predictor

    try:
        predictor = ChurnPredictor()
        logger.info("Predictor loaded successfully")
    except FileNotFoundError as error:
        logger.error("Failed to load predictor: %s", error)
        predictor = None


@app.get("/health")
def health() -> dict:
    model_loaded = predictor is not None

    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": "1.0.0",
        "threshold": predictor.threshold if predictor else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique os artefatos em models/.",
        )

    data = pd.DataFrame([payload.model_dump()])
    result = predictor.predict(data)

    return PredictionResponse(**result)