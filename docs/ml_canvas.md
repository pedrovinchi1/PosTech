# ML Canvas — Churn Predictor

## Problema de Negócio

Operadora de telecomunicações perde clientes em ritmo acelerado. Necessidade de identificar proativamente clientes com risco de cancelamento para acionar retenção.

## Stakeholders

- Diretoria de retenção
- Time de CRM

## Tipo de Problema

Classificação binária supervisionada — cliente vai cancelar? (Churn: Sim/Não)

## Dataset

- **Fonte:** IBM Telco Customer Churn
- **Volume:** 7.043 registros, 21 colunas
- **Target:** `Churn` (26.5% positivos — imbalance leve)
- **Período:** dados históricos de clientes ativos

## Métricas Técnicas

| Métrica | Justificativa |
|---------|--------------|
| AUC-ROC | Ranking geral de capacidade discriminativa |
| PR-AUC | Foco em positivos (churners) — mais informativo com desbalanceamento |
| F1 | Equilíbrio precisão/recall |

## Métricas de Negócio

- **Custo FN (não detectar churn):** R$500 — perde cliente, receita futura perdida
- **Custo FP (falso alarme):** R$50 — desconto de retenção desnecessário
- **Objetivo:** minimizar custo total via threshold ótimo

## SLOs

| SLO | Valor |
|-----|-------|
| Latência API (p99) | < 200ms |
| Disponibilidade | > 99% |

## Data Readiness

- [x] 7.043 registros suficientes para treino
- [x] 11 nulos em `TotalCharges` — todos `tenure=0` (comportamento esperado)
- [x] Sem leakage temporal
- [x] Sem features derivadas do target
- [x] Schema validado com pandera
- [x] `TotalCharges` corrigida de string para float

## Features

**Numéricas:** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`

**Categóricas:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

**Removida:** `customerID` (identificador sem sinal preditivo)

## Abordagem de Modelagem

1. Baselines: DummyClassifier, LogisticRegression, RandomForest, GradientBoosting, XGBoost
2. Modelo principal: MLP PyTorch (`hidden_dims=[128,64,32]`, dropout=0.3, early stopping)
3. Threshold ótimo determinado por curva de custo de negócio

## Deploy

- **Estratégia:** real-time (FastAPI + uvicorn)
- **Justificativa:** churn identificado individualmente a cada interação com CRM
- **Alternativa batch:** viável para campanhas de retenção mensais
