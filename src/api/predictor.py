import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.mlp import ChurnMLP

logger = logging.getLogger(__name__)


class ChurnPredictor:
    def __init__(
        self,
        model_path: str = "models/best_mlp.pt",
        preprocessor_path: str = "models/preprocessor.pkl",
        config_path: str = "models/model_config.json",
        threshold_path: str = "models/threshold.npy",
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.config_path = Path(config_path)
        self.threshold_path = Path(threshold_path)
        self.device = torch.device("cpu")

        self.preprocessor = self._load_preprocessor()
        self.threshold = self._load_threshold()
        self.model = self._load_model()

    def _load_preprocessor(self):
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor não encontrado em {self.preprocessor_path}"
            )

        return joblib.load(self.preprocessor_path)

    def _load_threshold(self) -> float:
        if not self.threshold_path.exists():
            logger.warning("Threshold não encontrado. Usando 0.5.")
            return 0.5

        return float(np.load(self.threshold_path))

    def _load_model(self) -> ChurnMLP:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config do modelo não encontrado em {self.config_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")

        with open(self.config_path, encoding="utf-8") as file:
            config = json.load(file)

        model = ChurnMLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        )

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def predict_proba(self, data: pd.DataFrame) -> float:
        X_processed = self.preprocessor.transform(data)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probability = torch.sigmoid(logits).cpu().numpy()[0]

        return float(probability)

    def predict(self, data: pd.DataFrame) -> dict:
        probability = self.predict_proba(data)

        return {
            "churn_probability": probability,
            "churn_prediction": bool(probability >= self.threshold),
            "threshold_used": self.threshold,
        }