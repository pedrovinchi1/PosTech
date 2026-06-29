# Model Card — Telco Customer Churn Predictor

## Visão geral
- **Tarefa:** classificação binária — prever se um cliente vai cancelar (`Churn`).
- **Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7.043 linhas, 21 colunas). Churn rate 26.5% (desbalanceado).
- **Versão do modelo:** 2.0.0
- **Modelo oficial:** MLP (PyTorch), arquitetura `[256, 128, 64]`, dropout 0.4, BatchNorm.
- **Modelos comparados:** LogisticRegression, RandomForest, GradientBoosting, XGBoost (tunado), MLP, e Ensemble (média XGBoost+MLP).

## Dados e features
- Split estratificado: train 5.633 / val 705 / test 705.
- `customerID` removido (identificador, sem sinal).
- `TotalCharges` convertido para float (8 nulos imputados pela mediana).
- **Feature engineering** (7 novas): `charges_per_month_active`, `monthly_to_total_ratio`, `tenure_x_monthly`, `num_services`, `tenure_group` (faixas), e `SeniorCitizen` tratado como categórico.
- Pré-processamento: `StandardScaler` (numéricas) + `OneHotEncoder` (categóricas) → 55 features finais.

## Treino
- Imbalance tratado com `pos_weight` (MLP / `BCEWithLogitsLoss`) e `scale_pos_weight` (XGBoost).
- MLP: AdamW, `lr=1e-3`, `weight_decay=1e-4`, early stopping por AUC de validação (paciência 20).
- XGBoost: `RandomizedSearchCV` (40 iterações, 5-fold, scoring `roc_auc`).
- Seed global = 42 (reprodutível).

## Métricas (test set, threshold escolhido na validação — sem leakage)

| Modelo | Ponto | Accuracy | F1 | AUC | PR-AUC |
|---|---|---|---|---|---|
| Ensemble | max-acc | 0.8128 | 0.6000 | 0.858 | 0.686 |
| Ensemble | balanced | 0.7929 | 0.6386 | 0.858 | 0.686 |
| **MLP** | **max-F1 (oficial)** | **0.7844** | **0.6498** | **0.857** | 0.671 |
| XGBoost | max-acc | 0.8099 | 0.5677 | 0.856 | 0.689 |

## Ponto de operação oficial
- **MLP, threshold = 0.575** (otimizado para F1 na validação).
- Resultado em test: **accuracy 78.4%, F1 0.650, AUC 0.857**.
- Justificativa: prioriza F1 (equilíbrio precision/recall na classe churn), métrica mais informativa que accuracy em dataset desbalanceado.

## Thresholds alternativos
- **Accuracy (≥80%):** usar Ensemble max-acc (acc 81.3%, F1 0.60).
- **Custo de negócio** (FN=500, FP=50): threshold ~0.28 (maximiza captura de churners; accuracy baixa por design).

## Limitações
- Teto de performance do dataset Telco Churn é ~80–82% accuracy / ~0.65 F1. Resultados acima disso indicam provável vazamento de dados.
- Modelo treinado em snapshot estático; sem validação temporal. Requer monitoramento de drift em produção.
- Não usar para decisões automáticas sem revisão humana (impacto em retenção de clientes).

## Uso pretendido
- Apoio ao time de retenção/CRM para priorizar clientes em risco.
- Servido via FastAPI (`POST /predict`). A API aplica `engineer_features` antes do pré-processamento.
