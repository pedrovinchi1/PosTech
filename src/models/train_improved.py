"""Improved training pipeline for Telco churn. Run: python -m src.models.train_improved"""
from __future__ import annotations
import json, logging, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import SEED
from src.data.loader import load_splits
from src.data.preprocessing import prepare_xy
from src.evaluation.metrics import COST_FN, COST_FP
from src.models.mlp import ChurnMLP

logger = logging.getLogger(__name__)
MIN_ACC = 0.80

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
ENGINEERED_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges",
                      "charges_per_month_active", "monthly_to_total_ratio",
                      "num_services", "tenure_x_monthly"]
ENGINEERED_CATEGORICAL = ["SeniorCitizen", "tenure_group", "gender", "Partner",
    "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]


def engineer_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    tenure_safe = df["tenure"].clip(lower=1)
    df["charges_per_month_active"] = df["TotalCharges"].fillna(0) / tenure_safe
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"].fillna(0) + 1.0)
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS if c in df.columns)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"]).astype(str)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    return df


def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, ENGINEERED_NUMERIC),
                              ("cat", categorical, ENGINEERED_CATEGORICAL)])


def pick_threshold(y_true, y_prob, objective="accuracy", min_acc=MIN_ACC):
    grid = np.linspace(0.05, 0.95, 181)
    if objective == "accuracy":
        return float(grid[int(np.argmax([accuracy_score(y_true, (y_prob >= t).astype(int)) for t in grid]))])
    if objective == "f1":
        return float(grid[int(np.argmax([f1_score(y_true, (y_prob >= t).astype(int)) for t in grid]))])
    if objective == "balanced":
        best_t, best_f1 = None, -1.0
        for t in grid:
            pred = (y_prob >= t).astype(int)
            if accuracy_score(y_true, pred) >= min_acc:
                f = f1_score(y_true, pred)
                if f > best_f1:
                    best_f1, best_t = f, float(t)
        if best_t is None:
            return float(grid[int(np.argmax([accuracy_score(y_true, (y_prob >= t).astype(int)) for t in grid]))])
        return best_t
    costs = []
    for t in grid:
        pred = (y_prob >= t).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        costs.append(fn * COST_FN + fp * COST_FP)
    return float(grid[int(np.argmin(costs))])


def metrics_at(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    return {"thr": round(float(thr), 3),
            "acc": round(accuracy_score(y_true, pred), 4),
            "f1": round(f1_score(y_true, pred), 4),
            "auc": round(roc_auc_score(y_true, y_prob), 4),
            "pr_auc": round(average_precision_score(y_true, y_prob), 4)}


def train_xgb(Xtr, ytr):
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    base = XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0,
                         scale_pos_weight=pos_weight, tree_method="hist")
    space = {"n_estimators": [200, 400, 600, 800], "max_depth": [3, 4, 5, 6],
             "learning_rate": [0.01, 0.02, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
             "reg_lambda": [1.0, 3.0, 5.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(base, space, n_iter=40, scoring="roc_auc", cv=cv,
                                random_state=SEED, n_jobs=-1, verbose=0)
    search.fit(Xtr, ytr)
    print("XGB best cv_auc=", round(search.best_score_, 4), search.best_params_)
    return search.best_estimator_


def train_mlp(Xtr, ytr, Xval, yval, input_dim, epochs=200, patience=20):
    torch.manual_seed(SEED)
    model = ChurnMLP(input_dim, hidden_dims=[256, 128, 64], dropout=0.4)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr.values, dtype=torch.float32)
    Xval_t = torch.tensor(Xval, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    best_auc, best_state, bad, last = 0.0, None, 0, 0
    for epoch in range(epochs):
        last = epoch
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(Xval_t)).numpy()
        auc = roc_auc_score(yval, val_prob)
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    print("MLP best val_auc=", round(best_auc, 4), "epoch", last + 1)
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_df, val_df, test_df = load_splits()
    Xtr_raw, ytr = prepare_xy(engineer_features(train_df))
    Xval_raw, yval = prepare_xy(engineer_features(val_df))
    Xte_raw, yte = prepare_xy(engineer_features(test_df))
    pre = build_preprocessor()
    Xtr = pre.fit_transform(Xtr_raw, ytr)
    Xval = pre.transform(Xval_raw)
    Xte = pre.transform(Xte_raw)
    print("n_features=", Xtr.shape[1])

    xgb = train_xgb(Xtr, ytr)
    xgb_val = xgb.predict_proba(Xval)[:, 1]
    xgb_te = xgb.predict_proba(Xte)[:, 1]
    mlp = train_mlp(Xtr, ytr, Xval, yval, Xtr.shape[1])
    mlp.eval()
    with torch.no_grad():
        mlp_val = torch.sigmoid(mlp(torch.tensor(Xval, dtype=torch.float32))).numpy()
        mlp_te = torch.sigmoid(mlp(torch.tensor(Xte, dtype=torch.float32))).numpy()
    ens_val = (xgb_val + mlp_val) / 2.0
    ens_te = (xgb_te + mlp_te) / 2.0

    probs = {"xgboost": (xgb_val, xgb_te), "mlp": (mlp_val, mlp_te), "ensemble": (ens_val, ens_te)}
    print("\n=== TEST metrics per model x operating-point (threshold picked on validation) ===")
    table = {}
    for name, (pv, pt) in probs.items():
        table[name] = {}
        for obj in ["accuracy", "f1", "balanced"]:
            thr = pick_threshold(yval, pv, obj)
            m = metrics_at(yte, pt, thr)
            table[name][obj] = m
            print(f"{name:9s} | {obj:9s} | thr={m['thr']:.3f} acc={m['acc']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} pr_auc={m['pr_auc']:.4f}")

    rec = table["ensemble"]["balanced"]
    print(f"\nRECOMMENDED: ensemble @ balanced -> acc={rec['acc']} f1={rec['f1']} (thr={rec['thr']})")

    with open("models/preprocessor_improved.pkl", "wb") as f:
        pickle.dump(pre, f)
    with open("models/best_xgb.pkl", "wb") as f:
        pickle.dump(xgb, f)
    torch.save(mlp.state_dict(), "models/best_mlp_improved.pt")
    thr_bal_xgb = pick_threshold(yval, xgb_val, "balanced")
    thr_bal_mlp = pick_threshold(yval, mlp_val, "balanced")
    thr_bal_ens = pick_threshold(yval, ens_val, "balanced")
    with open("models/operating_points.json", "w") as f:
        json.dump({"thresholds_balanced": {"xgboost": thr_bal_xgb, "mlp": thr_bal_mlp, "ensemble": thr_bal_ens},
                   "test_table": table}, f, indent=2)
    return table


if __name__ == "__main__":
    main()
