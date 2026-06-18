import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def evaluate(model: ChurnMLP, loader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item()
    return total_loss / len(loader)


def train(
    model: ChurnMLP,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs: int = 100,
    patience: int = 10,
    save_path: str = "models/best_mlp.pt",
) -> int:
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, criterion)
        logger.info("epoch", extra={"epoch": epoch + 1, "val_loss": round(val_loss, 4)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping", extra={"epoch": epoch + 1})
                break

    return epoch + 1
