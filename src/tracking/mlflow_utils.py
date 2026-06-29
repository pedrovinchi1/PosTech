import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd


def log_sklearn_experiment(
    model_name: str,
    model,
    params: dict,
    metrics: dict,
    X_train: pd.DataFrame,
) -> str:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id


def log_mlp_experiment(
    run_name: str,
    model,
    params: dict,
    metrics: dict,
) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, artifact_path="model")
        return run.info.run_id
