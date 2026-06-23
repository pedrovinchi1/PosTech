# Churn Predictor — Tech Challenge Fase 1

Projeto desenvolvido para o **Tech Challenge — Fase 1**, com o objetivo de construir um pipeline profissional de Machine Learning para previsão de churn de clientes de telecomunicações.

Rede neural (MLP PyTorch) para previsão de churn em telecomunicações, com pipeline end-to-end profissional.

O projeto utiliza o dataset **Telco Customer Churn** e implementa um fluxo end-to-end com:

* Análise exploratória dos dados;
* Pré-processamento com Scikit-learn;
* Modelos baseline;
* Rede Neural MLP com PyTorch;
* Análise de custo FP/FN e threshold ótimo;
* API de inferência com FastAPI;
* Validação de entrada com Pydantic;
* Testes automatizados com Pytest;
* Estrutura modular em `src/`.

---

## Objetivo do Projeto

O objetivo é prever se um cliente possui risco de churn, ou seja, se ele tende a cancelar o serviço.

A solução considera não apenas métricas técnicas, como AUC-ROC, PR-AUC e F1, mas também uma métrica de negócio baseada no custo de erros:

* **Falso negativo:** cliente que iria cancelar, mas o modelo não identificou;
* **Falso positivo:** cliente sinalizado como churn, mas que não cancelaria.

Como o falso negativo tem custo maior para o negócio, foi definido um threshold ótimo para reduzir o custo total esperado.

---

## Estrutura do Projeto

```text
.
├── data/
│   └── raw/
│       ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── docs/
│   ├── ml_canvas.md
│   ├── baselines_comparison.png
│   ├── baselines_cost_threshold.png
│   ├── baselines_feature_importance.png
│   ├── baselines_roc_pr_curves.png
│   ├── eda_churn_by_category.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_numeric_features.png
│   ├── eda_target_distribution.png
│   ├── eda_tenure_churn.png
│   ├── mlp_cost_threshold.png
│   ├── model_card.md
│   └── monitoring_plan.md
│    
├── models/
│   ├── best_mlp.pt
│   ├── preprocessor.pkl
│   ├── model_config.json
│   └── threshold.npy
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_mlp_experiments.ipynb
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── middleware.py
│   │   ├── predictor.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── mlp.py
│   │   └── train_mlp.py
│   └── tracking/
│       └── mlflow_utils.py
├── tests/
│   └── test_api.py
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```
O projeto possui:

* `src/data/` – loader + preprocessing pipeline
* `src/models/` – baselines (sklearn) + MLP (PyTorch)
* `src/evaluation/` – métricas AUC-ROC, PR-AUC, F1, custo negócio
* `src/tracking/` – wrappers MLflow
* `src/api/` – FastAPI + Pydantic + middleware
* `tests/` – smoke, schema (pandera), API
* `notebooks/` – EDA, baselines, experimentos MLP
* `docs/` – ML Canvas, Model Card, plano de monitoramento
* `data/raw/` – splits train/val/test
* `models/` – artefatos serializados (.pkl, .pt)
---

## Tecnologias Utilizadas

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* FastAPI
* Uvicorn
* Pydantic
* Pytest
* Ruff
* MLflow
* Matplotlib
* Seaborn
* XGBoost

---

## Setup
Também é possível usar `uv` para criar o ambiente e instalar dependências:

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

---

## Configuração do Ambiente

### 1. Criar ambiente virtual

Com Conda:

```bash
conda create -n pos python=3.11 -y
conda activate pos
```

Ou com `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 2. Instalar dependências

Via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ou via `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

---

## Dataset

O dataset utilizado é o **Telco Customer Churn**, disponível no arquivo:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

IBM Telco Customer Churn — 7.043 registros, 21 features, target binário (`Churn`).

Os dados foram separados em:

```text
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

A variável alvo é:

```text
Churn
```

Ela representa se o cliente cancelou ou não o serviço.

---

## Splits

| Split | Linhas | Churn% |
|-------|--------|--------|
| Train | 5.633  | 26.5%  |
| Val   | 705    | 26.5%  |
| Test  | 705    | 26.5%  |

---

## Métricas alvo

- AUC-ROC (ranking geral)
- PR-AUC (foco em positivos)
- F1
- Custo de negócio (FN=$500, FP=$50)

---

## Treinamento do Modelo

Para treinar a rede neural MLP:

```bash
make train-mlp
```

Ou diretamente:

```bash
python -m src.models.train_mlp
```

O treinamento gera os seguintes artefatos:

```text
models/best_mlp.pt
models/preprocessor.pkl
models/model_config.json
models/threshold.npy
```

### Artefatos gerados

* `best_mlp.pt`: pesos do melhor modelo MLP;
* `preprocessor.pkl`: pipeline de pré-processamento treinado;
* `model_config.json`: configuração da arquitetura da rede;
* `threshold.npy`: threshold ótimo para decisão de churn.

---

## Modelo MLP

A arquitetura principal utilizada foi uma rede neural MLP com PyTorch.

Configuração do modelo treinado:

```json
{
  "input_dim": 45,
  "hidden_dims": [128, 64, 32],
  "dropout": 0.3,
  "batch_size": 64,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "epochs_trained": 21
}
```

O threshold ótimo encontrado foi:

```text
0.35
```

A regra de decisão da API é:

```text
churn_prediction = churn_probability >= 0.35
```

---

## Rodar a API

Para iniciar a API:

```bash
make run-api
```

Ou diretamente:

```bash
uvicorn src.api.main:app --reload
```

A documentação interativa estará disponível em:

```text
http://127.0.0.1:8000/docs
```

---

## Endpoints

### Health Check

```http
GET /health
```

Exemplo com `curl`:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0",
  "threshold": 0.35,
  "timestamp": "2026-01-01T00:00:00+00:00"
}
```

---

### Predição de Churn

```http
POST /predict
```

Exemplo com `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0
  }'
```

Exemplo de resposta:

```json
{
  "churn_probability": 0.8327817320823669,
  "churn_prediction": true,
  "threshold_used": 0.35
}
```

Nesse exemplo, o modelo estimou aproximadamente **83,28% de probabilidade de churn**. Como o threshold definido é **0.35**, o cliente foi classificado como em risco de churn.

---

## Testes

Para executar os testes automatizados:

```bash
make test
```

Ou diretamente:

```bash
python -m pytest tests/ -v
```

Os testes validam:

* Funcionamento do endpoint `/health`;
* Funcionamento do endpoint `/predict`;
* Validação de schema inválido com Pydantic.

---

## Lint

Para verificar qualidade e padronização do código:

```bash
make lint
```

---

## Comandos

```bash
# Treinar baselines
make train-baselines

# Treinar MLP
make train-mlp

# Rodar API
make run-api

# Testes
make test

# Lint
make lint

# MLflow UI
.venv/Scripts/mlflow ui
```

---

## Makefile

Principais comandos disponíveis:

```bash
make train-mlp
make run-api
make test
make lint
```

---

## Métricas e Avaliação

O projeto utiliza métricas técnicas e de negócio:

* AUC-ROC;
* PR-AUC;
* F1-score;
* Custo de negócio baseado em falso positivo e falso negativo.

A função de custo considera:

```python
COST_FN = 500
COST_FP = 50
```

Isso reflete que deixar de identificar um cliente que vai cancelar é mais custoso do que abordar um cliente que não cancelaria.

---

## Status da Entrega

Principais entregas concluídas:

* EDA documentada;
* Pipeline de pré-processamento;
* Modelos baseline;
* MLP com PyTorch;
* Early stopping;
* Threshold ótimo por custo de negócio;
* API FastAPI funcional;
* Validação com Pydantic;
* Testes automatizados passando;
* Artefatos de modelo salvos;
* Inferência real via endpoint `/predict`.

---

## Próximos Passos

Possíveis melhorias futuras:

* Criar `model_card.md`;
* Criar `monitoring_plan.md`;
* Adicionar Docker;
* Adicionar GitHub Actions;
* Adicionar explicabilidade com SHAP;
* Adicionar monitoramento de drift com Evidently;
* Publicar a API em ambiente cloud.
