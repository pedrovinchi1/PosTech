import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_splits
from src.data.preprocessing import build_preprocessor, prepare_xy
from src.evaluation.metrics import compute_metrics, optimal_threshold
from src.models.mlp import ChurnMLP, train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
MODELS_DIR = Path("models")


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor_dataset(X, y) -> TensorDataset:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def predict_proba(model: ChurnMLP, X) -> np.ndarray:
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def main() -> None:
    set_seeds()
    MODELS_DIR.mkdir(exist_ok=True)

    logger.info("Loading train/val/test splits")
    train_df, val_df, test_df = load_splits()

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)

    logger.info("Building and fitting preprocessor")
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    input_dim = X_train_processed.shape[1]
    hidden_dims = [128, 64, 32]
    dropout = 0.3
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 100
    patience = 10

    logger.info(
        "Training configuration",
        extra={
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
        },
    )

    train_dataset = to_tensor_dataset(X_train_processed, y_train)
    val_dataset = to_tensor_dataset(X_val_processed, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    positives = y_train.sum()
    negatives = len(y_train) - positives
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)

    model = ChurnMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    actual_epochs = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        save_path="models/best_mlp.pt",
    )

    logger.info("Loading best model weights")
    model.load_state_dict(torch.load("models/best_mlp.pt", map_location="cpu"))
    model.eval()

    val_prob = predict_proba(model, X_val_processed)
    best_threshold = optimal_threshold(y_val, val_prob)

    test_prob = predict_proba(model, X_test_processed)
    test_metrics = compute_metrics(y_test, test_prob, threshold=best_threshold)

    logger.info("Test metrics", extra=test_metrics)

    logger.info("Saving preprocessor")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    logger.info("Saving threshold")
    np.save("models/threshold.npy", best_threshold)

    logger.info("Saving model config")
    model_config = {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_trained": actual_epochs,
    }

    with open("models/model_config.json", "w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=4)

    logger.info("Artifacts saved successfully")
    logger.info(
        "Final result",
        extra={
            "best_threshold": best_threshold,
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "f1": test_metrics["f1"],
            "business_cost": test_metrics["business_cost"],
        },
    )


if __name__ == "__main__":
    main()