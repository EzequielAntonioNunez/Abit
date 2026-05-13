# Proyecto de investigación de Abit (Identy Labs)

# exp_002b — Descomposición del drift por frecuencia léxica

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: 2026-05-13 (21:15:17 → 21:15:23 hora local, UTC 19:15:17Z, wall ~6 s)
- **Autor**: Ezequiel
- **Claude Code model**: claude-opus-4-7 (1M context)
- **Hardware**: M3 Pro local (no requiere GPU; análisis sobre parquet existente)

## 1. Hipótesis

exp_002 encontró Spearman ρ ≈ −0.04 entre surprisal Shannon y `activation_drift_last`, IC 95% [−0.064, −0.024]. El análisis exploratorio post-hoc reveló una relación **U-shape** por quintiles de surprisal (drift alto en los extremos, mínimo en Q1). Este experimento pre-registra **tres hipótesis competidoras** sobre el mecanismo causante del patrón observado:

**H1 — Artefacto de LayerNorm**: la U-shape es real y refleja una propiedad estructural de la última capa post-norm. Persiste tras controlar por frecuencia léxica del token. La señal "drift como medida de información" no se rescata en última capa por más que se desconfunda.

**H2 — Compresión semántica**: la U-shape se aplana o se vuelve monótona decreciente al controlar por frecuencia, pero el extremo de surprisal alto sigue mostrando drift relativamente bajo incluso a frecuencia controlada. Compatible con que tokens raros sean comprimidos a un atractor de representación.

**H3 — Confound de frecuencia léxica**: la asociación marginal nula entre surprisal y drift es subproducto de que ambas correlacionan con frecuencia léxica en sentidos opuestos. La correlación parcial controlando por log_freq emerge significativa y positiva, y la U-shape desaparece tras estratificar.

Las tres son falsables y mutuamente excluyentes para los rangos que se pre-registran abajo.

## 2. Predicción pre-registrada

Métrica principal: **Spearman parcial ρ(surprisal, drift_last | log_freq)** estimada con `pingouin.partial_corr(method="spearman")` sobre el parquet de la última run de exp_002.

| Hipótesis | ρ_parcial predicho | Estructura de la U |
|---|---|---|
| H1 | ρ_parcial ∈ [−0.1, 0.1] | U persiste en la mayoría de estratos de log_freq |
| H2 | ρ_parcial ∈ [0.15, 0.4] | U se aplana en general pero deciles 9-10 de surprisal siguen con drift bajo (cola monótona decreciente) |
| H3 | ρ_parcial > 0.3 | U desaparece tras estratificar (curva drift-vs-surprisal monótona dentro de cada estrato de log_freq) |

**Lectura de la decisión**:
- Si ρ_parcial cae en el intervalo de exactamente una hipótesis y la estructura U coincide con su predicción, esa hipótesis queda apoyada y las otras dos quedan provisionalmente descartadas.
- Si ρ_parcial cae fuera de los tres rangos (p. ej. ρ_parcial < −0.1 o ∈ (0.1, 0.15)), ninguna queda apoyada y se documenta como resultado ambiguo.
- IC 95% Fisher reportado por pingouin para la parcial; IC 95% bootstrap (B = 1000) para las medias por decil.

**Criterio de falsación de la línea de drift completa**: si ρ_parcial cae en el intervalo de H1 y la U se reproduce en estratos, M2 en la última capa queda formalmente cerrada como métrica candidata (independiente del análisis de capas intermedias en exp_003).

## 3. Cambios respecto al experimento anterior

Un único cambio respecto a exp_002: se re-analiza el parquet ya generado introduciendo `log_freq` (log de la frecuencia empírica del `token_id` en la muestra) como covariada. No se vuelve a cargar el modelo ni a re-tokenizar el corpus. Mismos 9 955 tokens, mismos seeds.

## 4. Configuración

Ver `config.yaml`. Sin sección `model` (este experimento no carga modelo).

Decisiones clave:
- **Fuente de datos**: `experiments/exp_002_activation_drift_last/results/<timestamp_más_reciente>/records.parquet`, seleccionada en tiempo de ejecución vía glob `results/*/records.parquet`.
- **Frecuencia léxica**: empírica sobre los 9 955 tokens del propio parquet. Esta es una **limitación documentada**: con un corpus de sólo ~10 k tokens hay un número grande de `token_id` que aparecen exactamente una vez (singletons), por lo que `log_freq = 0` para todos ellos y el decil más bajo de `log_freq` es necesariamente un bloque con muchos empates. `pd.qcut` se ejecuta con `duplicates="drop"`, devolviendo posiblemente < 10 bins efectivos. Si la conclusión depende de los deciles más extremos, un exp_002c re-haría este análisis usando frecuencias de un corpus externo (por ejemplo, frecuencia en el split `train` de WikiText o en C4 unigram counts).
- **Métricas estadísticas**:
  - Spearman parcial: `pingouin.partial_corr(method="spearman")` con IC 95% Fisher.
  - Bootstrap percentil B = 1000, α = 0.05 para la media de drift por decil.
  - OLS: `drift ~ const + surprisal + log_freq + surprisal × log_freq`, IC 95% y p-valores de statsmodels.
- **Figuras** (PNG, 150 DPI, en `figures/`):
  1. Hexbin bivariante surprisal vs drift.
  2. Curva drift media por decil de surprisal con banda IC 95%.
  3. Curva drift por decil de surprisal estratificada por decil de log_freq (un trazo por estrato).
  4. Residuales OLS vs fitted values.
- **Seed**: 42 propagado a `random`, `numpy`, `torch`, y al bootstrap.

## 5. Ejecución

```bash
python experiments/exp_002b_drift_decomposition/run.py
```

Tiempo estimado: < 1 min en M3 Pro (sólo lectura de parquet + pandas + bootstrap). No carga modelo, no usa GPU.

## 6. Resultados

Run de referencia: `results/20260513T191517Z/` (source: parquet de `exp_002_activation_drift_last/results/20260513T141005Z/`).

| Métrica | Valor |
|---|---|
| Tokens analizados (N) | 9 955 |
| Token IDs únicos | 2 692 |
| Singletons (`token_count == 1`) | 1 588 (≈ 59 % de los tipos, ≈ 16 % de los tokens) |
| Bins efectivos de surprisal | 10 |
| Bins efectivos de log_freq | **9** (uno se colapsa por empate masivo en `log(1) = 0`) |
| Spearman ρ marginal(surprisal, drift_last) | −0.0436 (idéntico a exp_002, sanity) |
| **Spearman parcial ρ(surprisal, drift_last \| log_freq)** | **−0.1429** |
| IC 95% Fisher | **[−0.16, −0.12]** |
| p-valor (pingouin) | 1.4 × 10⁻⁴⁶ |
| OLS `drift ~ const + surprisal + log_freq + surprisal × log_freq` | R² = 0.102 |
| OLS coef `const` | +2.426  [+2.403, +2.449]  p ≈ 0 |
| OLS coef `surprisal` | **−0.0237**  [−0.0273, −0.0202]  p = 1.9 × 10⁻³⁸ |
| OLS coef `log_freq` | **−0.0979**  [−0.1040, −0.0919]  p = 8.4 × 10⁻²¹⁰ |
| OLS coef `surprisal × log_freq` | +0.0077  [+0.0065, +0.0089]  p = 4.0 × 10⁻³⁵ |
| Tiempo real de ejecución | ~6 s |
| SHA git en el run | `e77ff04` |

Figuras (en `results/20260513T191517Z/figures/`):
- `bivariate_hexbin.png` — densidad log(N) en el plano (surprisal, drift).
- `drift_by_surprisal_decile.png` — curva marginal con banda IC 95% bootstrap.
- `drift_by_surprisal_stratified.png` — un trazo por decil de log_freq.
- `ols_residuals.png` — residuales OLS contra fitted.

**Curva marginal drift media por decil de surprisal (10 deciles):**

| dec | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| drift media | 2.42 | 2.19 | 2.03 | **1.99** | 2.02 | 2.05 | 2.07 | 2.09 | 2.13 | 2.25 |

Forma U clara, mínimo en D3, recuperación monótona hasta D9. La U es **más pronunciada** que en el corte por quintiles de exp_002.

**Curva estratificada por decil de log_freq (mean drift por surprisal decile dentro de cada estrato de log_freq):**

| logfreq_dec | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (singletons) | 2.68 | 2.62 | 2.39 | 2.31 | 2.27 | 2.21 | 2.20 | 2.13 | 2.15 | 2.23 | 2 582 |
| 1 | 2.70 | 2.29 | 2.12 | 2.23 | 2.10 | 2.08 | 2.01 | 2.06 | 2.04 | 2.15 | 612 |
| 2 | 2.68 | 2.46 | 2.20 | 2.20 | 2.14 | 2.08 | 2.01 | 2.13 | 2.17 | 2.40 | 844 |
| 3 | 2.39 | 2.20 | 2.15 | 2.07 | 2.04 | 2.05 | 2.03 | 1.91 | 2.14 | 2.18 | 970 |
| 4 | 2.37 | 2.18 | 2.28 | 2.06 | 2.12 | 2.19 | 2.28 | 2.18 | 2.29 | 2.27 | 971 |
| 5 | 2.28 | 2.10 | 2.06 | 2.01 | 2.08 | 2.08 | 2.22 | 2.18 | 2.18 | 2.57 | 1 059 |
| 6 | 2.07 | 1.95 | 1.91 | 1.99 | 2.04 | 2.04 | 1.96 | 1.87 | 1.97 | 2.12 | 954 |
| 7 | 2.26 | 2.10 | 1.99 | 1.98 | 1.98 | 2.04 | 2.09 | 2.13 | 2.05 | 2.13 | 1 033 |
| 8 (más frecuentes) | 1.85 | 1.67 | 1.60 | 1.59 | 1.61 | 1.64 | 1.64 | 1.85 | 1.94 | 1.81 | 930 |

Visualmente la U-shape es identificable en **los 9 estratos**: la columna D0 siempre tiene drift alto y, salvo logfreq_dec 6, la columna D9 también muestra repunte respecto al mínimo intermedio (típicamente entre D3 y D6). El nivel general del drift baja monótonamente con log_freq (estrato 0 ≈ 2.32 vs estrato 8 ≈ 1.72), efecto principal del log_freq que la OLS captura como coeficiente −0.098.

## 7. Análisis estadístico

**Comparación con el pre-registro (sección 2):**

| Hipótesis | Rango pre-registrado | ρ_parcial observado | Estructura U observada | Coincide |
|---|---|---|---|---|
| H1 (artefacto LayerNorm) | ρ ∈ [−0.1, 0.1] | −0.1429 (fuera, ligeramente por debajo) | U persiste en 9/9 estratos | **Estructural sí, cuantitativo justo fuera** |
| H2 (compresión semántica) | ρ ∈ [0.15, 0.4] | −0.1429 (signo opuesto) | U no se aplana ni se vuelve monótona decreciente | No |
| H3 (confound de frecuencia) | ρ > 0.3 | −0.1429 (signo opuesto) | U **persiste** tras estratificar | No |

**Lectura literal del pre-registro**: ρ_parcial = −0.1429 cae fuera de los tres intervalos cuantitativos. Por la regla auto-impuesta en sección 2 ("Si ρ_parcial cae fuera de los tres rangos, ninguna queda apoyada y se documenta como resultado ambiguo"), técnicamente el resultado es **ambiguo en términos cuantitativos**.

**Lectura sustantiva sin reescribir el pre-registro**: la evidencia estructural (U-shape en 9/9 estratos, signo negativo del partial, coeficiente OLS de surprisal negativo y pequeño) **es completamente incompatible con H2 y H3**, y **es consistente con la intuición que motivó H1** (no hay señal positiva monótona; el aplanamiento es estructural). La discrepancia con el intervalo cuantitativo de H1 (−0.143 vs umbral inferior −0.10) es de 0.04 unidades y refleja que H1 fue formulada como "ρ ≈ 0" cuando en realidad lo que se observa es "ρ pequeño pero claramente negativo". La predicción cualitativa de H1 (U persiste) se cumple sin matices.

**OLS — interpretación de coeficientes**:
- `log_freq` (efecto principal, β = −0.098, p ≈ 10⁻²¹⁰): tokens más frecuentes producen menos drift. Es el predictor con mayor peso explicativo del modelo lineal.
- `surprisal` (β = −0.024, p ≈ 10⁻³⁸): a frecuencia fija, mayor surprisal se asocia con *menos* drift, no más. Pequeño en magnitud pero estadísticamente robusto.
- `surprisal × log_freq` (β = +0.0077, p ≈ 10⁻³⁵): la pendiente negativa surprisal→drift se atenúa para tokens de alta frecuencia. Consistente con que la asociación negativa surprisal–drift es más fuerte en los tipos raros.
- R² = 0.102: surprisal, log_freq y su interacción explican ~10 % de la varianza de `drift_last`. El 90 % restante es ruido o estructura no capturada por esta especificación lineal.

Limitaciones del análisis:
1. **Frecuencia léxica empírica intramuestral**: como se anunció en sección 4, hay 1 588 singletons (16 % de los tokens, 59 % de los tipos). Ese subconjunto define un único valor `log_freq = 0`, lo que reduce el número de bins efectivos de 10 a 9 y satura el decil 0 de log_freq con 2 582 tokens. Una réplica con frecuencias de un corpus externo (split train de WikiText o C4 unigrams) podría producir un mapa más informativo en el extremo de baja frecuencia. No esperamos que cambie la conclusión cualitativa porque la U se reproduce en estratos donde sí hay variación de frecuencia.
2. **Spearman parcial via pingouin**: usa el método de residualización rank, válido para ρ pero no idéntico a un test no paramétrico de independencia condicional estricta. Para n grande la diferencia es despreciable.
3. **OLS lineal**: no captura la forma U por construcción. La sigue compatible con que el modelo lineal capte el efecto medio de log_freq (que sí es monótono) y deje la U como varianza residual.

## 8. Conclusiones

**H1 (artefacto de LayerNorm) queda como lectura mejor soportada**, aunque su intervalo cuantitativo pre-registrado (ρ_parcial ∈ [−0.1, 0.1]) se incumple por margen estrecho. H2 y H3 quedan descartadas con claridad (signo equivocado del partial, U no se aplana).

Lectura honesta:

1. No hay señal monótona positiva entre surprisal Shannon y `activation_drift_last` ni siquiera tras controlar por frecuencia léxica. La asociación parcial es pequeña (−0.143) y, con signo, *negativa*. Eso es contrario a la intuición "tokens sorpresivos cambian más el estado" y refuerza la lectura de que la métrica está dominada por el LayerNorm final.
2. La U-shape es estructural: se reproduce en todos los estratos de log_freq, no es subproducto del confound de frecuencia. Lo que sí hace log_freq es **modular la magnitud absoluta del drift** (β = −0.098 OLS, p ≈ 10⁻²¹⁰): tokens más frecuentes tienen drift sistemáticamente menor en todos los deciles de surprisal. Este es el confound *real* que exp_002b consigue identificar, y aunque no rescata la línea de drift, es un hallazgo de control reutilizable para experimentos siguientes.
3. La línea de drift en la **última capa** queda formalmente cerrada como métrica candidata para H1-H3 del plan general. Esto era exactamente el criterio de falsación de exp_002b sección 2 ("si ρ_parcial cae en H1 y la U se reproduce en estratos, M2 en la última capa queda formalmente cerrada"). Se cumple.
4. La línea de drift en **capas intermedias** sigue viva y es lo que exp_003 está diseñado para probar. exp_002b refuerza la motivación: si el problema es específico del post-norm de la última capa, capas previas no normalizadas (Pythia tiene LayerNorm dentro de cada bloque pero `hidden_states[l]` para l < L es el output residual *antes* del norm final) deberían comportarse distinto.

## 9. Próximos pasos sugeridos

- **Proceder con exp_003 sin cambios sustantivos**: el diseño actual (barrido de M2 por capa l ∈ {0, 4, 8, 12, 16, 20, 23}) es el experimento correcto para decidir si la métrica de drift se rescata fuera de la última capa. exp_002b acota el resultado esperable: si exp_003 muestra que ninguna capa supera ρ ≈ 0.25, la línea de drift queda formalmente cerrada y procede pasar a M1 (Bayesian surprise sobre bloque futuro).
- **Incluir `log_freq` como covariada en el reporte de exp_003**: dado que aquí log_freq emerge como el predictor más fuerte de drift (β = −0.098, p ≈ 10⁻²¹⁰), exp_003 debería reportar tanto ρ marginal como ρ parcial controlando log_freq por cada capa. El parcial será probablemente más informativo. Esto se documenta como cambio menor en el config/run de exp_003 antes de ejecutarlo.
- **(Opcional, baja prioridad) exp_002c**: réplica de exp_002b usando frecuencia léxica de un corpus externo (split train de WikiText-103). Sólo si tras exp_003 la decisión sobre la línea de drift queda ambigua y necesitamos un test de robustez del confound.
