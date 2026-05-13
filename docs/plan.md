# Proyecto de investigación de Abit (Identy Labs)

# Plan de Investigación

**Título de trabajo**: Información como Transformación — Más allá de Shannon en Modelos de Lenguaje

**Autor**: Ezequiel
**Versión**: 0.1 (draft inicial)
**Estado**: pre-registro

---

## 1. Resumen ejecutivo

Investigar empíricamente si una métrica de información basada en el **cambio estructural inducido en un modelo** (sorpresa significativa) predice mejor el impacto cognitivo/funcional de un token que la entropía clásica de Shannon, usando LLMs open-source como sistema observador.

La pregunta no es filosófica: es medible. Si dos tokens con surprisal Shannon idénticos producen cambios internos drásticamente distintos, Shannon es incompleto para sistemas inteligentes.

---

## 2. Hipótesis

### H1 — Divergencia Shannon vs. impacto interno
Existen pares de tokens `(t_a, t_b)` en contexto `C` tales que:

```
| -log p(t_a|C) - (-log p(t_b|C)) | < ε
```

pero el cambio inducido en las representaciones internas del modelo (activaciones, attention, distribución predictiva del siguiente bloque) difiere en órdenes de magnitud.

**Predicción falsable**: si para N ≥ 10.000 pares emparejados por surprisal, la varianza del activation drift normalizada por surprisal es ≤ 0.1, H1 se rechaza.

### H2 — Bayesian surprise sobre bloque futuro
La KL-divergencia entre la distribución predictiva del modelo sobre los siguientes `k` tokens antes y después de observar `t_i`:

```
S_B = D_KL( p(t_{i+1:i+k} | C, t_i) || p(t_{i+1:i+k} | C) )
```

correlaciona mejor con tareas downstream (cloze, QA factual, sentiment shift) que el surprisal clásico, controlando por surprisal.

**Predicción falsable**: si el coeficiente parcial de S_B en regresión múltiple con surprisal como covariada no es significativo (p > 0.01) en ≥ 2 de 3 modelos, H2 se rechaza.

### H3 — Métrica de deformación representacional
Una métrica basada en la deformación del espacio de activaciones capa-por-capa (no solo la última) captura un componente de "información" ortogonal a Shannon y a Bayesian surprise.

**Predicción falsable**: si la varianza explicada residual por activation drift, tras controlar por M0 y M1, es < 5%, H3 se rechaza.

---

## 3. Preguntas de investigación

1. ¿Es el surprisal Shannon un predictor débil del cambio funcional interno en LLMs?
2. De N métricas candidatas, ¿cuál correlaciona mejor con cambios downstream observables?
3. ¿Existen métricas irreducibles entre sí o todas colapsan a variantes de KL?
4. ¿Es estable el ranking de métricas entre arquitecturas y escalas (1B–7B)?

---

## 4. Por qué este problema importa ahora

- El campo de **mechanistic interpretability** (Anthropic, EleutherAI) ya mide activaciones internas rutinariamente. La infraestructura existe.
- **Active inference** (Friston) y **Bayesian surprise** (Itti & Baldi) ya plantean parte de la teoría, pero rara vez se validan en LLMs modernos.
- El **information bottleneck** (Tishby) muestra que la "información relevante" es definible operacionalmente.
- Hay un hueco real: nadie ha comparado sistemáticamente Shannon vs. métricas de cambio interno en LLMs sobre datasets controlados.

---

## 5. Fases del proyecto

| Fase | Duración | Entregable |
|------|----------|------------|
| 0. Setup y replicación de baselines | 1 semana | Pipeline funcional, surprisal Shannon replicado en WikiText |
| 1. Implementación de métricas candidatas | 2 semanas | M0–M5 implementadas, tests unitarios, validación numérica |
| 2. Experimentos en corpus controlado | 2 semanas | Resultados en WikiText, LAMBADA, BLiMP |
| 3. Validación en tareas downstream | 1 semana | Resultados en Counterfact, QA |
| 4. Análisis estadístico y replicación cross-model | 1 semana | Tablas finales, intervalos de confianza |
| 5. Escritura y decisión paper/preprint | 1 semana | Decisión go/no-go para preprint |

**Total estimado**: 8 semanas a ritmo de proyecto paralelo (10–15 h/semana). Reescalable a 4 semanas full-time.

---

## 6. Criterios de éxito

### Mínimo (publicable como nota técnica)
Caracterizar empíricamente la divergencia entre Shannon y al menos una métrica alternativa en LLMs reales, con análisis estadístico riguroso.

### Esperado (preprint corto)
Identificar al menos una métrica con correlación parcial r > 0.4 con impacto downstream que Shannon no capture, replicada en ≥ 2 modelos.

### Aspiracional (paper completo)
Formular una definición operacional reproducible de "sorpresa significativa" con propiedades teóricas demostrables (no-negatividad, invarianza ante reparametrización, relación con métricas existentes).

---

## 7. Criterios de fracaso (pre-registro honesto)

- Si todas las métricas correlacionan r > 0.9 con Shannon en todos los modelos, la hipótesis original es trivial.
- Si los resultados aparecen solo en un modelo o un dataset, no son resultados.
- Si la métrica propuesta no es reproducible con seeds fijos, no existe.

Estos casos terminan el proyecto. Se publicará el resultado negativo si ocurre.

---

## 8. Riesgos identificados

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| Confound de frecuencia léxica | Alta | Estratificación por log-frecuencia |
| Coste computacional excede M3 | Media | Modelos ≤ 3B; cloud puntual (Runpod, Modal) |
| Métricas triviales (colapsan a Shannon) | Media | Análisis de varianza residual obligatorio |
| Reinvención de literatura existente | Alta | Revisión bibliográfica formal en Fase 0 |
| Sesgo de confirmación al elegir tokens | Alta | Selección aleatoria estratificada, no curada |

---

## 9. Estado del arte mínimo a revisar (Fase 0)

- Shannon, C. (1948). *A Mathematical Theory of Communication*
- Itti, L. & Baldi, P. (2009). *Bayesian Surprise Attracts Human Attention*
- Friston, K. (2010). *The Free-Energy Principle*
- Hahn, M., Jurafsky, D., Futrell, R. (2021). *Sensitivity as a complexity measure for sequence classification tasks*
- Meng, K. et al. (2022). *Locating and Editing Factual Associations in GPT* (Counterfact)
- Tishby, N. & Zaslavsky, N. (2015). *Deep Learning and the Information Bottleneck Principle*
- Saphra, N. & Lopez, A. — *Understanding learning dynamics of LMs*
- Papers recientes de mechanistic interpretability (Anthropic 2023–2025)

---

## 10. Ética y reproducibilidad

- Todos los datasets son públicos, licencias compatibles con investigación.
- Todos los modelos son open-weights (Gemma, Llama, Pythia).
- Código MIT, datos y resultados versionados con DVC o equivalente.
- Pre-registro de hipótesis y predicciones cuantitativas antes de correr experimentos finales.
- Reporte obligatorio de resultados negativos.

---

## Enmiendas

### 2026-05-13 — Modelo base de Fase 0

`google/gemma-2-2b`, listado como modelo principal en este plan, se sustituye por `EleutherAI/pythia-1.4b` como modelo base de las Fases 0-3. Gemma 2 2B pasa a Fase 4 (replicación cross-architecture). Motivación, alternativas y consecuencias en `docs/decisions/0002-modelo-base-pythia.md`. Los criterios de éxito y los rangos cuantitativos pre-registrados no se modifican.

### 2026-05-13 — Cierre de la línea de drift geométrico (M2)

La métrica M2 (`activation drift` en norma L2 normalizada) queda cerrada como candidata primaria para evaluar las hipótesis H1-H3 de este plan tras tres pre-registros falsificados consecutivamente (exp_002 en última capa, exp_002b tras controlar log_freq, exp_003 barriendo 8 capas en Pythia 1.4B). La métrica primaria pasa a ser M1 (Bayesian surprise sobre bloque futuro), evaluada en exp_004 contra la tarea downstream T1 (cloze). Razones, alternativas y consecuencias en `docs/decisions/0004-cierre-linea-drift-geometrico.md`. La estructura de las hipótesis H1-H3 no se reformula; lo que cambia es la métrica candidata que las instancia.
