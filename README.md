# Churn Predictor — Tech Challenge Fase 1

Pipeline end-to-end de Machine Learning para **previsão de churn** em telecomunicações.
Modelo principal: **MLP (PyTorch)**. Baselines em Scikit-Learn, tracking com MLflow, servido via FastAPI.

## Dataset

IBM Telco Customer Churn — 7.043 registros, 21 features, target binário (`Churn`). Taxa de churn 26,5% (desbalanceado).

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

## Estrutura

```
src/
  data/          # loader + preprocessing pipeline (sklearn)
  models/        # baselines (sklearn) + MLP (PyTorch) + train_improved (pipeline final)
  evaluation/    # métricas AUC-ROC, PR-AUC, F1, custo de negócio
  tracking/      # wrappers MLflow
  api/           # FastAPI + Pydantic + middleware (serve o modelo final)
tests/           # smoke, schema (pandera), API
notebooks/       # 01_eda, 02_baselines, 03_mlp_experiments
docs/            # ML Canvas, Model Card, plano de monitoramento, roteiro do vídeo
data/raw/        # splits train/val/test
models/          # artefatos serializados (.pkl, .pt, thresholds, operating_points.json)
```

## Comandos

```bash
# Treinar baselines (Dummy, LogReg, RandomForest, GradientBoosting, XGBoost) + MLflow
make train-baselines

# Pipeline final: feature engineering + XGBoost tunado + MLP + ensemble
.venv/Scripts/python.exe -m src.models.train_improved

# Rodar API
make run-api          # -> http://127.0.0.1:8000/docs

# Testes (smoke, schema, API)
make test

# Lint
make lint

# MLflow UI
make mlflow-ui        # -> http://127.0.0.1:5000  (sqlite:///mlflow.db)
```

## Pipeline final (`src/models/train_improved.py`)

Melhorias sobre o baseline ingênuo:
- **Feature engineering** (7 novas): `tenure_group`, `charges_per_month_active`, `monthly_to_total_ratio`, `tenure_x_monthly`, `num_services`; `SeniorCitizen` tratado como categórico.
- **Tratamento de desbalanceamento:** `pos_weight` (MLP / `BCEWithLogitsLoss`) e `scale_pos_weight` (XGBoost).
- **Tuning:** `RandomizedSearchCV` (40 iterações, 5-fold) no XGBoost.
- **Ensemble:** média das probabilidades XGBoost + MLP.
- **Seleção de threshold na validação** (sem leakage no teste), com 3 pontos de operação.

## Resultados (test set — threshold escolhido na validação)

| Modelo | Ponto | Accuracy | F1 | AUC | PR-AUC |
|---|---|---|---|---|---|
| Ensemble | max-acc | 0.813 | 0.600 | 0.858 | 0.686 |
| Ensemble | balanced | 0.793 | 0.639 | 0.858 | 0.686 |
| **MLP** | **max-F1 (oficial)** | **0.784** | **0.650** | **0.857** | 0.671 |
| XGBoost | max-acc | 0.810 | 0.568 | 0.856 | 0.689 |

> **Sobre accuracy vs F1:** as duas métricas competem pelo threshold em base desbalanceada.
> O dataset tem teto conhecido de ~80–82% accuracy / ~0,65 F1 — números muito acima disso indicam vazamento.
> Os pontos de operação ficam em `models/operating_points.json`; detalhes e justificativa no [Model Card](docs/model_card.md).

## Splits

| Split | Linhas | Churn% |
|-------|--------|--------|
| Train | 5.633  | 26.5%  |
| Val   | 705    | 26.5%  |
| Test  | 705    | 26.5%  |

## API

`POST /predict` recebe os campos do cliente (validados por Pydantic), aplica `engineer_features` + pré-processamento e retorna probabilidade + decisão no threshold oficial.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":89.5,"TotalCharges":1070.0}'
# -> {"churn_probability": ..., "churn_prediction": ..., "threshold_used": 0.575}
```

Artefatos servidos: `best_mlp_improved.pt`, `preprocessor_improved.pkl`, `threshold_mlp_f1.npy`.

## Entregáveis

- ✅ EDA + [ML Canvas](docs/ml_canvas.md)
- ✅ Baselines + MLflow tracking
- ✅ MLP PyTorch + análise de custo FP/FN (FN=$500, FP=$50)
- ✅ `src/` modular + FastAPI + Pydantic + testes + Makefile + ruff
- ✅ [Model Card](docs/model_card.md) + [Roteiro do vídeo STAR](docs/roteiro_video.md)
