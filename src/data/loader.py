import logging

import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("Loaded dataset", extra={"rows": len(df), "cols": len(df.columns)})
    return df


def load_splits(base_dir: str = "data/raw") -> tuple[pd.DataFrame, ...]:
    train = pd.read_csv(f"{base_dir}/train.csv")
    val = pd.read_csv(f"{base_dir}/val.csv")
    test = pd.read_csv(f"{base_dir}/test.csv")
    logger.info(
        "Loaded splits",
        extra={"train": len(train), "val": len(val), "test": len(test)},
    )
    return train, val, test
