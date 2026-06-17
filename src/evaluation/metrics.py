import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

COST_FN = 500
COST_FP = 50


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "business_cost": business_cost(y_true, y_pred),
        "threshold": threshold,
    }


def business_cost(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fp * COST_FP + fn * COST_FN)


def optimal_threshold(y_true, y_prob, thresholds=None) -> float:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    costs = [business_cost(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])
