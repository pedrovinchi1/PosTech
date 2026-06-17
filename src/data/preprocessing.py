import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) e target (y) de um DataFrame de churn.

    Nos splits o target ``Churn`` já está codificado como 0/1; ``customerID``
    já foi removido. ``TotalCharges`` pode conter strings vazias, então é
    convertido para numérico (o imputer do pipeline trata os NaN resultantes).
    """
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    logger.info("Prepared X/y", extra={"rows": len(X), "features": X.shape[1]})
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Constrói o pipeline de pré-processamento sklearn (numérico + categórico)."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
