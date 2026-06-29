import logging
import pickle

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src import SEED
from src.data.loader import load_splits
from src.data.preprocessing import build_preprocessor, prepare_xy
from src.tracking.mlflow_utils import log_sklearn_experiment

logger = logging.getLogger(__name__)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = ["roc_auc", "average_precision", "f1"]

BASELINES = {
    "dummy": DummyClassifier(strategy="most_frequent", random_state=SEED),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    "xgboost": XGBClassifier(
        n_estimators=100, random_state=SEED, eval_metric="logloss", verbosity=0
    ),
}


def run_baselines():
    train_df, val_df, test_df = load_splits()

    preprocessor = build_preprocessor()
    X_train, y_train = prepare_xy(train_df)
    X_train_proc = preprocessor.fit_transform(X_train, y_train)

    results = {}
    for name, model in BASELINES.items():
        logger.info("Training baseline", extra={"model": name})
        scores = cross_validate(
            model, X_train_proc, y_train,
            cv=CV, scoring=SCORING, return_train_score=True,
        )
        metrics = {
            "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "pr_auc_mean": float(np.mean(scores["test_average_precision"])),
            "f1_mean": float(np.mean(scores["test_f1"])),
            "fit_time_mean": float(np.mean(scores["fit_time"])),
        }
        model.fit(X_train_proc, y_train)
        log_sklearn_experiment(name, model, {"model": name}, metrics, X_train)
        results[name] = metrics
        logger.info("Baseline done", extra={"model": name, **metrics})

    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_baselines()
