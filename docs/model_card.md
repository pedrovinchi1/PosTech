# Model Card — Churn MLP

## Descrição

MLP binário (PyTorch) para classificação de risco de churn em telecomunicações.  
Arquitetura: 3 camadas ocultas [256, 128, 64] + BatchNorm + Dropout(0.3). Threshold otimizado por custo de negócio (FN=R$500, FP=R$50).

## Dataset

- **Fonte:** IBM Telco Customer Churn (Kaggle / IBM Sample Data)
- **Volume total:** 7.043 registros | 21 features
- **Split:** 80% treino (5.634) / 10% val (704) / 10% teste (705) — estratificado
- **Taxa de churn:** ~26,5% (desbalanceado)

## Performance — Conjunto de Teste

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.8459 |
| PR-AUC | 0.6552 |
| F1-Score | 0.5665 |
| Precisão | 0.4022 |
| Recall | 0.9572 |
| Acurácia | 0.6113 |
| Custo de negócio | R$ 17.300 |
| Threshold ótimo | 0.23 |

> **Nota sobre threshold:** threshold baixo (0.23) é intencional — o custo de falso negativo (R$500/cliente perdido) é 10× maior que o de falso positivo (R$50/campanha desnecessária). O modelo privilegia recall alto para minimizar clientes churned não detectados.

## Comparação com Baselines (test set, threshold=0.5 para baselines)

| Modelo | AUC-ROC | PR-AUC | F1 |
|--------|---------|--------|----|
| Dummy (most_frequent) | 0.0000 | 0.2652 | 0.0000 |
| LogisticRegression | 0.8528 | 0.6575 | 0.5957 |
| RandomForest | 0.8203 | 0.6193 | 0.5394 |
| GradientBoosting | 0.8510 | 0.6775 | 0.5938 |
| XGBoost | 0.8294 | 0.6510 | 0.5808 |
| **MLP (best, thresh=0.23)** | **0.8459** | **0.6552** | **0.5665** |

> MLP é competitivo com os melhores baselines (LogReg, GBM) em AUC-ROC e PR-AUC. Com threshold otimizado (0.23), obtém recall de 95,7% — superior a qualquer baseline no threshold padrão 0.5.

## Limitações

- Treinado em dados de uma operadora específica
- Pode não generalizar para outros perfis de telecom
- Desbalanceamento leve (~26% positivos)
- Sem dados temporais explícitos (modelo estático)

## Vieses Identificados

- Taxa de churn varia por `gender` e `SeniorCitizen`
- Clientes com contrato `Month-to-month` têm churn muito superior

## Cenários de Falha

- Dados faltantes em `TotalCharges` sem imputação prévia
- Features fora do range de treino (ex: `tenure > 72`)
- Drift significativo em `MonthlyCharges` ou `tenure`

## Monitoramento

Ver `docs/monitoring_plan.md`
