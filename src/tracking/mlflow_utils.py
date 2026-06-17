import mlflow
import mlflow.pytorch
import mlflow.sklearn
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
        # cloudpickle evita a validação de "trusted types" do skops (mlflow 3.x),
        # que rejeita modelos como o XGBClassifier por padrão.
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )
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
