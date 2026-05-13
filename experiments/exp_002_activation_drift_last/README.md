# Proyecto de investigación de Abit (Identy Labs)

# exp_002 — Activation drift, última capa

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: 2026-05-13 (17:14:47 → 17:15:31 hora local, UTC 15:14:47Z, wall ~44 s)
- **Autor**: Ezequiel
- **Claude Code model**: claude-opus-4-7 (1M context)
- **Hardware**: M3 Pro local, backend `mps`, dtype `bfloat16`

## 1. Hipótesis

El surprisal Shannon y la deriva de activación en la última capa (`M2_last`, definido en `docs/design.md` §3) miden cosas correlacionadas pero **no redundantes** en Pythia 1.4B sobre WikiText-103. Es decir: la deriva en la última capa contiene una señal positivamente asociada al surprisal — al cabo, ambos dependen de cómo de inesperado es el token — pero no es una mera reescritura de él.

## 2. Predicción pre-registrada

- **Correlación Spearman ρ(surprisal_bits, activation_drift_last)** ∈ [0.3, 0.6].
- **IC 95% bootstrap (B = 1000)**: no contiene 0 (la métrica no es ruido) ni 0.85 (no es redundante con Shannon).

**Criterios de falsación explícitos**:
- Si ρ > 0.85 o el IC inferior > 0.85: las métricas son efectivamente redundantes en la última capa. La hipótesis H3 del `plan.md` (existe una componente de información ortogonal a Shannon capturada por drift) queda comprometida en la última capa; habría que mover M2 a capas intermedias en exp_003.
- Si ρ < 0.2 o el IC superior < 0.2: probablemente hay un bug en la métrica o en el alineamiento de tokens (la asociación física esperada entre "token sorpresivo" y "cambio interno grande" no aparece a ningún nivel relevante). Investigar antes de continuar.
- Si ρ < 0 con IC que no incluye 0: relación inesperada, parar y depurar.

## 3. Cambios respecto al experimento anterior

Un único cambio respecto a exp_001: se añade la métrica `activation_drift_last` calculada sobre los mismos tokens (mismo modelo, mismo seed, mismo dataset, mismo skip de warmup) y se reporta la correlación Spearman entre ambas métricas con IC bootstrap. No se modifica el sampling ni el subconjunto de documentos.

## 4. Configuración

Ver `config.yaml`.

Decisiones clave:
- Modelo: `EleutherAI/pythia-1.4b` en bfloat16 (consolidado como base de Fase 0 por ADR `docs/decisions/0002-modelo-base-pythia.md`).
- Dataset y filtros idénticos a exp_001 (WikiText-103 validation, ≥ 200 chars, ≤ 100 docs, contextos de 256 tokens, skip 50 primeros).
- Seed 42 propagado a `random`, `numpy`, `torch`, y al bootstrap.
- Bootstrap Spearman: B = 1000, α = 0.05 (IC 95% percentil).
- Métrica `M2_last` definida exactamente como en `docs/design.md` §3:
  `|| h_L(t_1..t_i) - h_L(t_1..t_{i-1}) ||_2 / sqrt(d_L)`.
  Implementada en `src/iat/metrics/activation.py`, exportada vía `iat.metrics`.

**Nota de eficiencia (justificación para no aumentar el tiempo estimado)**: aunque el diseño habla de "comparar antes y después de observar `t_i`", el masking causal de los LMs autoregresivos hace que ambos estados ocultos relevantes — `h_L(C, t_i)` y `h_L(C)` — estén disponibles en el output de **un único forward pass** sobre la secuencia completa: son las filas `i` e `i-1` de `hidden_states[-1]` respectivamente. Por eso `activation_drift_last_layer` consume sólo un forward pass extra con `output_hidden_states=True`, no |C| forwards.

## 5. Ejecución

```bash
python experiments/exp_002_activation_drift_last/run.py
```

Tiempo estimado: similar a exp_001 (~13 min wall, dominados por carga de pesos y dataset; ~15-30 s adicionales por documento por el `output_hidden_states=True`). No se proyecta más allá de 20 min en M3 Pro.

Si la memoria de M3 se satura (no esperado, hidden_states del último layer son ~1 MB por doc), reducir `max_docs` aquí — no en exp_001.

## 6. Resultados

Run de referencia: `results/20260513T151447Z/`

| Métrica | Valor |
|---|---|
| Modelo / device / dtype | `EleutherAI/pythia-1.4b` / `mps` / `torch.bfloat16` |
| Documentos usados | 98 (idénticos a exp_001) |
| Tokens evaluados (N) | 9 955 (idénticos a exp_001) |
| Surprisal Shannon (bits/token) | media 3.639, mediana 2.306, p95 11.564 — **idénticos a exp_001** (sanity check de alineación) |
| Activation drift last (norma L2 / √d) | media 2.124, mediana 2.089, p95 3.078, σ 0.499 |
| Spearman ρ(surprisal, drift_last) | **−0.0436** |
| IC 95% bootstrap (B = 1000) | **[−0.0640, −0.0235]** |
| p-valor (scipy.stats.spearmanr) | 1.36 × 10⁻⁵ |
| Pearson r (referencia) | +0.0380 |
| Tiempo real de ejecución | ~44 s (modelo en caché; sin coste extra de descarga) |
| SHA git en el run | `af39d12` |

Artefactos en el run: `config.snapshot.yaml`, `summary.json`, `records.parquet` (~183 KB, 5 columnas, 9955 filas), `git_sha.txt`, `env.txt`.

## 7. Análisis estadístico

Comparación contra el pre-registro (sección 2):

| Predicción | Rango pre-registrado | Valor observado | ¿Se cumple? |
|---|---|---|---|
| Spearman ρ ∈ [0.3, 0.6] | [0.3, 0.6] | −0.0436 | **No** |
| IC 95% no contiene 0 | excluye 0 | IC [−0.064, −0.024]: excluye 0 (por debajo) | Excluye 0, pero del lado equivocado |
| IC 95% no contiene 0.85 | excluye 0.85 | IC excluye 0.85 trivialmente | Sí (irrelevante en este escenario) |

**Cuál criterio de falsación se ha disparado**

Sección 2 enumeraba tres criterios de falsación:

1. *"ρ > 0.85 → métricas redundantes, mover M2 a capas intermedias"*: **no se dispara** (ρ es casi cero, no alto).
2. *"ρ < 0.2 → probablemente hay un bug"*: **se dispara** (ρ = −0.04, IC superior −0.024 ≪ 0.2).
3. *"ρ < 0 con IC que no incluye 0 → relación inesperada, parar y depurar"*: **se dispara** (IC = [−0.064, −0.024], estrictamente negativo).

Pre-registro respetado: no se ajusta la predicción a posteriori.

**Sanity checks adicionales realizados (post-hoc, marcados como tales)**

Antes de etiquetar el resultado como "bug" se inspecciona la estructura del registro `records.parquet` con análisis exploratorio (NO confirmatorio):

- La columna `surprisal_bits` y los estadísticos descriptivos coinciden exactamente con exp_001 (media 3.639146 vs 3.639146, mediana 2.305774 vs 2.305774). La alineación token-a-token entre las dos métricas está garantizada por construcción y empíricamente verificada.
- La columna `activation_drift_last` no es degenerada: σ = 0.499, rango [0.536, 3.852], distribución unimodal aproximadamente simétrica alrededor de 2.1.
- Quintiles de surprisal vs. media de drift (post-hoc):

  | Quintil de surprisal | drift media | drift mediana | n |
  |---|---|---|---|
  | Q0 (más bajo) | 2.304 | 2.255 | 1991 |
  | Q1 | 2.011 | 1.986 | 1991 |
  | Q2 | 2.035 | 2.026 | 1991 |
  | Q3 | 2.078 | 2.067 | 1991 |
  | Q4 (más alto) | 2.191 | 2.166 | 1991 |

  La relación es **no monótona, en forma de U**: tokens muy predecibles (Q0) y tokens muy sorpresivos (Q4) tienen drift relativamente alto; el mínimo está en Q1. Spearman, por construcción, captura mal una asociación U-shape, lo que explica naturalmente ρ ≈ 0 con un p-valor pequeño (la asociación existe, pero no es monótona).

  El minimum en Q1 sugiere que tokens *moderadamente comunes* (function words, BPE-pieces) producen el menor cambio de estado interno, mientras que los extremos de la distribución de surprisal — tanto los muy frecuentes (artículos, signos de puntuación al inicio de oración) como los muy raros (nombres propios, términos técnicos) — perturban más el último hidden state. Esto **no estaba pre-registrado**, es exploratorio.

No se realiza inferencia formal multi-test sobre los quintiles: es exploratorio y se reporta como tal, en línea con `docs/design.md` §8.

## 8. Conclusiones

**Hipótesis falsada en su forma pre-registrada.** La predicción de ρ ∈ [0.3, 0.6] entre surprisal Shannon y `activation_drift_last` en Pythia 1.4B sobre WikiText-103 **no se cumple**: ρ = −0.0436 con IC 95% [−0.064, −0.024], estadísticamente significativo del lado equivocado. Dos de los tres criterios de falsación se disparan.

Lectura honesta del resultado, sin reescribir la hipótesis:

1. La asociación monótona positiva esperada **no existe** en la última capa de Pythia 1.4B. La metáfora "tokens más sorpresivos producen mayor cambio en la última capa" no se sostiene empíricamente, al menos no en sentido lineal-monótono.

2. La métrica `activation_drift_last_layer` no está degenerada (varía y se distribuye razonablemente). El alineamiento token-a-token con `shannon_surprisal` es exacto (mismos descriptivos que exp_001). Por tanto **no es un bug numérico evidente**.

3. La exploración post-hoc por quintiles sugiere una relación **no monótona en forma de U**, no anticipada por el pre-registro. Esto es interpretable a la luz de que `hidden_states[-1]` en transformers autoregresivos es el output del LayerNorm/RMSNorm final: por construcción está normalizado para ser leído por el unembedding, lo que aplana la magnitud del drift entre posiciones consecutivas y borra la señal lineal. El patrón U podría reflejar un efecto residual donde los tokens en los extremos (muy o poco predecibles) son los que más perturban la geometría incluso tras el norm final.

4. La pregunta científica de fondo (¿captura M2 algo ortogonal a Shannon?) **no queda contestada por este experimento**. Lo único que queda contestado es que la **versión last-layer** de M2, evaluada con Spearman, no es un sustituto monótono de Shannon. Compatible con que M2 en capas intermedias sí lo sea — el diseño original lo predecía como rama alternativa.

5. No se observa ningún signo de problema en el pipeline (perplejidad, distribución de surprisal, tiempos de ejecución, alineación token-a-token). El veredicto es metodológico-conceptual, no de implementación.

## 9. Próximos pasos sugeridos

El resultado bloquea la rama "feliz" pero abre dos ramas concretas, en este orden de prioridad:

- **exp_003 — Activation drift en capas intermedias (M2_l, l ∈ {6, 12, 18}; n_layers de Pythia 1.4B = 24)**. Predicción a pre-registrar antes de ejecutar: al menos una capa intermedia produce Spearman ρ con surprisal en [0.3, 0.7], y la curva ρ(l) tiene un máximo *antes* de la última capa. Si esto se confirma, valida que la normalización final es el culpable del nulo en exp_002 y abre la métrica M2 para Fase 2. Si todas las capas dan ρ < 0.2, M2 en su forma actual queda en cuestión y procede pasar a M1 (Bayesian surprise) antes que persistir con drift.
- **exp_002b — re-evaluación de M2_last con una métrica no-monótona** (p. ej. mutual information no paramétrica vía bins, o Kendall sobre rangos absolutos `|surprisal − mediana|`) sobre los mismos records ya guardados. Coste casi cero: usa el parquet de este run, no requiere correr el modelo. Sirve para cuantificar el patrón U-shape detectado en el análisis exploratorio sin re-ejecutar.

Ambas extensiones son experimentos nuevos (numeración separada) y no modifican exp_002. exp_002 queda como histórico inmutable: hipótesis pre-registrada, falsada, documentada.
