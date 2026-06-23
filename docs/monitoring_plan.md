# Plano de Monitoramento — Churn Predictor

## Métricas de Modelo

| Métrica | Frequência | Alerta |
|---------|-----------|--------|
| AUC-ROC em batch | Mensal | Queda > 5% → retreino |
| F1 em batch | Mensal | Queda > 10% → retreino |
| Taxa de churn previsto | Semanal | Desvio > 15% da baseline |

## Métricas de Dados (Data Drift)

| Feature | Método | Threshold |
|---------|--------|-----------|
| `MonthlyCharges` | PSI | PSI > 0.2 → alerta |
| `tenure` | PSI | PSI > 0.2 → alerta |
| `Contract` | Chi-quadrado | p < 0.05 → alerta |

## Métricas de API

| Métrica | SLO | Alerta |
|---------|-----|--------|
| Latência p99 | < 200ms | > 200ms → investigar |
| Error rate | < 1% | > 1% → PagerDuty |
| Throughput | > 100 req/s | < 50 req/s → escalar |

## Playbook de Resposta

1. **Alerta dispara** → verificar dashboard de drift de dados
2. **Drift confirmado** → coletar novos dados do período de drift
3. **Se drift leve** → re-calibrar threshold sem retreinar
4. **Se drift severo** → retreinar modelo com dados recentes
5. **Validar no conjunto de holdout** → AUC-ROC ≥ modelo anterior
6. **Deploy da nova versão** → via MLflow Model Registry (staging → production)
7. **Monitorar por 48h** → confirmar estabilização das métricas

## Ferramentas

- **Drift:** Evidently AI (relatório HTML mensal)
- **API metrics:** uvicorn logs + middleware de latência
- **Experimentos:** MLflow tracking + Model Registry
- **Alertas:** (a definir por infraestrutura do cliente)
