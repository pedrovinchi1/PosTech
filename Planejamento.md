# Planejamento — Tech Challenge Fase 1
## Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End

> Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7.043 linhas, 21 colunas, target: `Churn`)

---

## Estrutura de Diretórios (alvo final)

```
.
├── data/
│   ├── raw/                        # CSV original (nunca modificar)
│   └── processed/                  # artefatos de pré-processamento
├── docs/
│   ├── ml_canvas.md                # ML Canvas preenchido
│   ├── model_card.md               # Model Card completo
│   └── monitoring_plan.md          # Plano de monitoramento
├── models/                         # artefatos serializados (.pkl, .pt)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_mlp_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # carrega e valida CSV
│   │   └── preprocessing.py        # pipeline sklearn + transformadores custom
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py            # DummyClassifier, LogisticRegression, RandomForest, XGBoost
│   │   └── mlp.py                  # MLP PyTorch (arquitetura + loop de treino)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # AUC-ROC, PR-AUC, F1, custo FP/FN
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── mlflow_utils.py         # wrappers de log MLflow
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app
│       ├── schemas.py              # Pydantic models
│       └── middleware.py           # logging latência
├── tests/
│   ├── test_smoke.py               # smoke test: pipeline roda sem erro
│   ├── test_schema.py              # pandera: schema do CSV
│   └── test_api.py                 # endpoints /predict e /health
├── pyproject.toml                  # single source of truth
├── Makefile                        # lint, test, run, train
├── .gitignore
└── README.md
```

---

## Fase 1 — Setup e Estrutura do Projeto

**Objetivo:** repositório funcional antes de qualquer código de ML.

### 1.1 Inicialização do repositório
- [ ] Criar estrutura de diretórios acima
- [ ] Mover CSV para `data/raw/`
- [ ] Criar `.gitignore` adequado para ML:
  - `*.pkl`, `*.pt`, `mlruns/`, `__pycache__/`, `.env`, `data/processed/`, `*.egg-info/`

### 1.2 pyproject.toml
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "churn-predictor"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "scikit-learn>=1.4",
    "mlflow>=2.11",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pydantic>=2.6",
    "pandas>=2.2",
    "numpy>=1.26",
    "pandera>=0.18",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "httpx>=0.27",
    "ruff>=0.4",
    "ipykernel",
]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 1.3 Makefile
```makefile
lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

train-baselines:
	python -m src.models.baselines

train-mlp:
	python -m src.models.mlp

run-api:
	uvicorn src.api.main:app --reload

all: lint test
```

### 1.4 Seeds globais
- Definir constante `SEED = 42` em `src/__init__.py`
- Fixar em todos os módulos: `random.seed`, `np.random.seed`, `torch.manual_seed`

---

## Fase 2 — EDA e Preparação dos Dados (Etapa 1 do Challenge)

**Objetivo:** entender profundamente o dataset, documentar achados, validar qualidade.

### 2.1 ML Canvas (`docs/ml_canvas.md`)
- Stakeholders: diretoria de retenção, time de CRM
- Problema: classificação binária — cliente vai cancelar?
- Métricas de negócio: custo de churn evitado (custo de FN >> custo de FP)
- Métricas técnicas: AUC-ROC (ranking), PR-AUC (foco em positivos), F1
- SLOs: latência API < 200ms, disponibilidade > 99%
- Data readiness: 7.043 registros, 11 nulos em `TotalCharges`

### 2.2 EDA — `notebooks/01_eda.ipynb`

**Análise univariada:**
- Distribuição de `Churn` (26% positivos — imbalance leve, não exige SMOTE obrigatório)
- Histogramas para `tenure`, `MonthlyCharges`, `TotalCharges`
- Barplots para todas categóricas

**Análise bivariada:**
- Churn rate por `Contract` (mês-a-mês vs anual vs 2 anos)
- Churn rate por `InternetService`, `TechSupport`, `OnlineSecurity`
- Correlação de `tenure` e `MonthlyCharges` com churn
- Heatmap de correlação (numéricas)

**Qualidade:**
- 11 nulos em `TotalCharges` — investigar: todos com `tenure == 0` → imputar com 0 ou dropar
- `customerID` — drop (identificador, sem sinal preditivo)
- `TotalCharges` como string no CSV — converter para float

**Data readiness checklist:**
- [ ] Sem leakage temporal
- [ ] Sem features derivadas do target
- [ ] Schema validado com pandera

### 2.3 Pré-processamento — `src/data/preprocessing.py`

Pipeline sklearn reprodutível:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
])
```

### 2.4 Split estratificado
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
```

---

## Fase 3 — Baselines (Etapa 1 do Challenge)

**Objetivo:** estabelecer floor de performance, registrar no MLflow.

### 3.1 Modelos baseline — `src/models/baselines.py`

| Modelo | Justificativa |
|--------|--------------|
| `DummyClassifier(strategy='most_frequent')` | Floor absoluto |
| `LogisticRegression` | Baseline linear interpretável |
| `RandomForestClassifier` | Baseline não-linear (ensemble) |
| `GradientBoostingClassifier` ou `XGBClassifier` | Baseline boosting |

### 3.2 Validação cruzada estratificada
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_validate(
    model, X_train, y_train, cv=cv,
    scoring=['roc_auc', 'average_precision', 'f1'],
    return_train_score=True
)
```

### 3.3 MLflow tracking — `src/tracking/mlflow_utils.py`
```python
import mlflow

def log_sklearn_experiment(model_name, model, params, metrics, X_train, y_train):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_input(mlflow.data.from_pandas(X_train), context="training")
        mlflow.sklearn.log_model(model, artifact_path="model")
```

**Métricas a logar:** `roc_auc_mean`, `roc_auc_std`, `pr_auc_mean`, `f1_mean`, `fit_time_mean`

### 3.4 Entregável Fase 3
- `notebooks/02_baselines.ipynb` com tabela comparativa
- Runs registrados em `mlruns/`

---

## Fase 4 — MLP com PyTorch (Etapa 2 do Challenge)

**Objetivo:** modelo principal, superar baselines, análise de custo.

### 4.1 Arquitetura MLP — `src/models/mlp.py`

```python
import torch
import torch.nn as nn

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
```

**Configuração inicial:** `hidden_dims=[128, 64, 32]`, `dropout=0.3`

### 4.2 Loop de treinamento com early stopping

```python
def train(model, optimizer, criterion, train_loader, val_loader,
          epochs=100, patience=10):
    best_val_loss = float('inf')
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_mlp.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
```

**Detalhes:**
- Loss: `BCEWithLogitsLoss` (com `pos_weight` para imbalance)
- Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)`
- Batch size: 64
- Split: 80% train / 10% val / 10% test

### 4.3 Hyperparameter tuning (básico)
Registrar no MLflow as seguintes variações:
- `hidden_dims`: `[64,32]`, `[128,64,32]`, `[256,128,64]`
- `dropout`: `0.2`, `0.3`, `0.5`
- `lr`: `1e-3`, `5e-4`

### 4.4 Análise de custo FP/FN

```python
# Custo de negócio
COST_FN = 500   # não detectar churn → perde cliente (custo alto)
COST_FP = 50    # falso alarme → desconto desnecessário (custo baixo)

def business_cost(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_FP + fn * COST_FN
```

- Plotar curva de custo vs threshold
- Encontrar threshold ótimo que minimiza custo de negócio

### 4.5 Tabela comparativa de modelos

| Modelo | AUC-ROC | PR-AUC | F1 | Custo Negócio |
|--------|---------|--------|----|---------------|
| Dummy | - | - | - | - |
| LogReg | - | - | - | - |
| RandomForest | - | - | - | - |
| GradientBoosting | - | - | - | - |
| MLP (best) | - | - | - | - |

### 4.6 MLflow para MLP
```python
with mlflow.start_run(run_name="mlp_v1"):
    mlflow.log_params({
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "epochs_trained": actual_epochs,
    })
    mlflow.log_metrics({"roc_auc": auc, "pr_auc": pr_auc, "f1": f1})
    mlflow.pytorch.log_model(model, "model")
```

---

## Fase 5 — Engenharia e API (Etapa 3 do Challenge)

**Objetivo:** código production-ready, API funcional, testes passando.

### 5.1 Refatoração em módulos
- Todo código de notebooks migra para `src/`
- Sem `print()` — usar `logging` estruturado:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Training started", extra={"epoch": epoch, "loss": loss})
```

### 5.2 FastAPI — `src/api/main.py`

**Endpoints obrigatórios:**

`GET /health`
```json
{"status": "ok", "model_version": "1.0.0", "timestamp": "..."}
```

`POST /predict`
```json
// Request
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "tenure": 12,
  "MonthlyCharges": 65.5,
  ...
}
// Response
{
  "churn_probability": 0.73,
  "churn_prediction": true,
  "threshold_used": 0.42
}
```

### 5.3 Pydantic schemas — `src/api/schemas.py`
```python
from pydantic import BaseModel, Field
from typing import Literal

class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    # ... demais features
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float
```

### 5.4 Middleware de latência
```python
import time
from fastapi import Request

@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("request", extra={
        "path": request.url.path,
        "method": request.method,
        "latency_ms": round(latency_ms, 2),
        "status_code": response.status_code,
    })
    return response
```

### 5.5 Testes — `tests/`

**test_smoke.py** — pipeline de pré-processamento roda sem erro com dados sintéticos

**test_schema.py** — validação pandera do CSV de entrada
```python
import pandera as pa

schema = pa.DataFrameSchema({
    "tenure": pa.Column(int, pa.Check.ge(0)),
    "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
    "TotalCharges": pa.Column(float, nullable=True),
    "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
})
```

**test_api.py** — endpoints com `httpx.AsyncClient`
```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid():
    payload = {...}  # exemplo válido
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert 0 <= r.json()["churn_probability"] <= 1

def test_predict_invalid_schema():
    r = client.post("/predict", json={"tenure": -1})
    assert r.status_code == 422
```

---

## Fase 6 — Documentação e Entrega Final (Etapa 4 do Challenge)

### 6.1 Model Card — `docs/model_card.md`
- **Descrição:** MLP binário para classificação de churn
- **Dataset:** Telco Customer Churn IBM, 7.043 registros
- **Performance:** AUC-ROC, PR-AUC, F1 (test set)
- **Threshold:** valor ótimo por custo de negócio
- **Limitações:**
  - Treinado em dados de uma operadora específica
  - Pode não generalizar para outros perfis de telecom
  - Desbalanceamento leve (~26% positivos)
- **Vieses identificados:** taxa de churn varia por `gender` e `SeniorCitizen`
- **Cenários de falha:** dados faltantes em `TotalCharges`, features fora do range de treino
- **Monitoramento:** drift em `MonthlyCharges` e `tenure`

### 6.2 Plano de monitoramento — `docs/monitoring_plan.md`
- **Métricas de modelo:** AUC-ROC em batch mensal, F1, taxa de churn previsto
- **Métricas de dados:** distribuição de `tenure`, `MonthlyCharges` (PSI < 0.2)
- **Métricas de API:** latência p99 < 200ms, error rate < 1%
- **Alertas:** AUC-ROC cair > 5% → retreino automático
- **Playbook de resposta:**
  1. Alert dispara → verificar drift de dados
  2. Se drift confirmado → coletar novos dados e retreinar
  3. Deploy da nova versão com MLflow model registry

### 6.3 Arquitetura de deploy — `docs/model_card.md` (seção deploy)
- **Escolha:** real-time (FastAPI + uvicorn)
- **Justificativa:** churn é identificado individualmente a cada interação com CRM
- **Alternativa batch:** viável para campanhas de retenção mensais
- **(Opcional) Cloud:** containerizar com Docker → deploy no Cloud Run (GCP) ou App Service (Azure)

### 6.4 README.md final
```markdown
## Setup
pip install -e ".[dev]"

## Treinar modelos
make train-baselines
make train-mlp

## Rodar API
make run-api

## Testes
make test

## Lint
make lint

## MLflow UI
mlflow ui
```

### 6.5 Vídeo STAR (5 min)
- **Situation (1min):** operadora perde clientes, precisa de predição de churn
- **Task (1min):** pipeline end-to-end com MLP PyTorch + API FastAPI
- **Action (2min):** decisões técnicas — arquitetura MLP, threshold por custo, FastAPI
- **Result (1min):** métricas obtidas, lições aprendidas, próximos passos

---

---

## Fase 7 — BÔNUS: Diferenciais Além do Pedido

> Nada aqui é obrigatório. Cada item adiciona pontuação implícita e impressiona avaliadores.

### 7.1 Explicabilidade com SHAP

Adicionar interpretabilidade ao modelo — responder "por que este cliente vai cancelar?".

```python
# pip install shap
import shap

# Para o MLP PyTorch (via DeepExplainer)
explainer = shap.DeepExplainer(model, X_train_tensor[:100])
shap_values = explainer.shap_values(X_test_tensor[:50])

# Top features que mais contribuem para churn
shap.summary_plot(shap_values, X_test_df, plot_type="bar")
```

**Entregável:**
- `notebooks/04_explainability.ipynb` com SHAP summary plot e force plot de casos individuais
- Endpoint `/explain` na API retornando top-5 features para cada predição:

```json
// POST /explain
{
  "churn_probability": 0.81,
  "top_factors": [
    {"feature": "Contract_Month-to-month", "impact": +0.32},
    {"feature": "tenure", "impact": -0.21},
    {"feature": "TechSupport_No", "impact": +0.18}
  ]
}
```

---

### 7.2 Dashboard de Monitoramento com Evidently

Relatório automático de data drift e model performance em produção.

```python
# pip install evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=X_train_df, current_data=X_new_df,
           column_mapping=column_mapping)
report.save_html("docs/drift_report.html")
```

**Entregável:**
- Script `src/monitoring/drift_report.py` que gera relatório HTML
- Makefile target: `make drift-report`
- Relatório de exemplo em `docs/drift_report.html`

---

### 7.3 Containerização com Docker

API completamente containerizada, pronta para qualquer cloud.

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e "."

COPY src/ src/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./mlruns:/app/mlruns
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0
```

**Entregável:**
- `Dockerfile` + `docker-compose.yml`
- Makefile targets: `make docker-build`, `make docker-run`
- README com instruções Docker

---

### 7.4 CI/CD com GitHub Actions

Pipeline automático de lint + testes a cada push.

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: pytest tests/ -v --tb=short
```

**Entregável:**
- `.github/workflows/ci.yml`
- Badge de CI no README: `![CI](https://github.com/user/repo/actions/workflows/ci.yml/badge.svg)`

---

### 7.5 MLflow Model Registry + Versionamento

Promover modelos via staging → production usando MLflow Registry.

```python
import mlflow.tracking

client = mlflow.tracking.MlflowClient()

# Registrar modelo
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-mlp"
)

# Transição staging → production
client.transition_model_version_stage(
    name="churn-mlp",
    version=result.version,
    stage="Production"
)
```

**Entregável:**
- API carrega modelo diretamente do Registry:
  ```python
  model = mlflow.pytorch.load_model("models:/churn-mlp/Production")
  ```
- Seção no README sobre versionamento de modelos

---

### 7.6 Feature Engineering Avançada

Features derivadas que aumentam performance do modelo.

| Feature nova | Fórmula | Intuição |
|---|---|---|
| `charges_per_month` | `TotalCharges / (tenure + 1)` | Consistência de gasto |
| `service_count` | soma de serviços ativos | Engajamento com produto |
| `is_new_customer` | `tenure <= 3` | Novos clientes churn mais |
| `high_value` | `MonthlyCharges > 75` | Clientes premium |
| `no_support_services` | `TechSupport == No AND OnlineSecurity == No` | Perfil de risco |

```python
# src/data/feature_engineering.py
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
    service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count'] = (df[service_cols] == 'Yes').sum(axis=1)
    df['is_new_customer'] = (df['tenure'] <= 3).astype(int)
    df['high_value'] = (df['MonthlyCharges'] > 75).astype(int)
    return df
```

**Entregável:**
- `src/data/feature_engineering.py`
- Ablation study no notebook: modelo com vs. sem features derivadas
- MLflow comparando runs com e sem feature engineering

---

### 7.7 Teste de Carga na API

Validar que a API aguenta volume real com `locust`.

```python
# tests/load_test.py
from locust import HttpUser, task, between

class ChurnAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", json={
            "gender": "Female", "SeniorCitizen": 0,
            "tenure": 12, "MonthlyCharges": 65.5,
            # ...
        })

    @task(3)
    def health(self):
        self.client.get("/health")
```

```bash
# Makefile
load-test:
	locust -f tests/load_test.py --headless -u 50 -r 10 --run-time 60s --host http://localhost:8000
```

**Entregável:**
- `tests/load_test.py`
- Relatório de resultados: throughput, latência p50/p95/p99

---

### Resumo do Impacto dos Bônus

| Item | Esforço | Impacto | O que impressiona |
|------|---------|---------|------------------|
| SHAP + /explain | Médio | Alto | Mostra visão de produto, não só ML |
| Evidently drift | Baixo | Médio | MLOps real |
| Docker + compose | Baixo | Alto | Deploy profissional |
| GitHub Actions CI | Baixo | Alto | Engenharia de software séria |
| MLflow Registry | Baixo | Médio | Gestão de ciclo de vida |
| Feature engineering | Médio | Alto | Pode melhorar AUC-ROC 2-5% |
| Teste de carga | Baixo | Médio | Prova que a API é robusta |

**Prioridade recomendada:** Docker → CI/CD → SHAP → Feature Engineering → Evidently

---

## Critérios de Avaliação × Fases

| Critério | Peso | Fase |
|----------|------|------|
| Rede neural PyTorch | 25% | Fase 4 |
| Código e estrutura | 20% | Fase 1, 5 |
| Pipeline e reprodutibilidade | 15% | Fase 2, 3 |
| API de inferência | 15% | Fase 5 |
| Documentação e Model Card | 10% | Fase 6 |
| Vídeo STAR | 10% | Fase 6 |
| Bônus deploy nuvem | 5% | Fase 6 |

---

## Checklist Final de Entrega

- [ ] Repositório GitHub com estrutura organizada
- [ ] README.md completo com setup e execução
- [ ] pyproject.toml como single source of truth
- [ ] Commits limpos e significativos
- [ ] .gitignore adequado para ML
- [ ] EDA completa em notebook
- [ ] ML Canvas preenchido
- [ ] Baselines registrados no MLflow
- [ ] MLP PyTorch funcional com early stopping
- [ ] Tabela comparativa ≥ 4 métricas
- [ ] Análise de custo FP/FN com threshold ótimo
- [ ] Pipeline sklearn reprodutível com seeds
- [ ] FastAPI com /predict e /health
- [ ] Validação Pydantic
- [ ] Logging estruturado (sem print())
- [ ] Middleware de latência
- [ ] ≥ 3 testes automatizados (smoke, schema, API)
- [ ] ruff sem erros
- [ ] Model Card completo
- [ ] Plano de monitoramento
- [ ] Vídeo STAR 5 minutos
- [ ] (Opcional) Deploy em nuvem com URL pública
