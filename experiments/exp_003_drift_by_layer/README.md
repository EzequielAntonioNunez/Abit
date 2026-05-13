# Proyecto de investigación de Abit (Identy Labs)

# exp_003 — Activation drift, barrido por capa

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: 2026-05-13 (22:12:30 → 22:13:14 hora local, UTC 20:12:30Z, wall ~44 s)
- **Autor**: Ezequiel
- **Claude Code model**: claude-opus-4-7 (1M context)
- **Hardware**: M3 Pro local, backend `mps`, dtype `bfloat16`

## 1. Hipótesis

El nulo en la última capa observado en exp_002 (Spearman ρ ≈ −0.04) y reforzado por exp_002b (ρ_parcial = −0.143 tras controlar log_freq, U-shape persistente en 9/9 estratos) **es específico de la última capa post-norm**, no una propiedad estructural de toda la representación interna del modelo. En consecuencia, **existe al menos una capa intermedia l\*** de Pythia 1.4B en la que la norma L2 normalizada del cambio por posición (M2_l\*, definición en `docs/design.md` §3) **muestra una asociación monótona positiva no trivial con el surprisal Shannon**.

Si esta hipótesis es cierta, la línea de drift como métrica de información sigue viva (deja exp_002b como acotado al caso post-norm). Si no, queda formalmente cerrada y procede pivotar a M1 (Bayesian surprise sobre bloque futuro) como métrica candidata principal.

## 2. Predicción pre-registrada

**Métrica primaria**: el máximo sobre las 8 capas barridas de la correlación Spearman marginal entre `surprisal_bits` y `drift_l` sobre los mismos 9 955 tokens de exp_001/exp_002.

- **Predicción principal**: `max_l ρ_marginal(surprisal, drift_l) >= 0.25` con IC 95% bootstrap cuyo límite inferior > 0.10.
- **Predicción de forma**: el argmax del máximo cae en una capa intermedia (l ∈ {8, 12, 16}), no en los extremos (l = 0, 20, 23, -1). La curva ρ(l) tiene un máximo único o una meseta en capas medias.
- **Predicción complementaria post-exp_002b**: el ρ_parcial(surprisal, drift_l | log_freq) en la capa argmax cumple `ρ_parcial >= 0.20`, asegurando que la señal no es subproducto del confound de frecuencia léxica que exp_002b identificó (β_log_freq ≈ −0.098 sobre drift_last).

**Criterio de falsación (cierre de la línea de drift)**: si `max_l ρ_marginal < 0.25` y `max_l ρ_parcial < 0.20`, la línea de drift como métrica de información queda formalmente cerrada para Pythia 1.4B sobre WikiText. exp_004 abandona drift y arranca M1 (Bayesian surprise).

**Criterio de falsación intermedio (poco probable pero pre-registrado)**: si max ρ > 0.7, la métrica colapsa a Shannon en alguna capa, lo que sería una redundancia indicativa de que M2 no aporta información ortogonal. Documentar y plantear cómo afecta a H3 del plan general.

## 3. Cambios respecto al experimento anterior

Un único cambio respecto a exp_002: el barrido pasa de una capa (la última) a **8 capas**: `l ∈ {0, 4, 8, 12, 16, 20, 23, -1}`. Las primeras 7 son índices absolutos sobre `outputs.hidden_states` (que en Pythia 1.4B tiene longitud 25); el último `-1` se incluye explícitamente para reproducir el valor de exp_002 con la misma cadena de cómputo y permitir verificación numérica de coherencia.

Cambios menores derivados de exp_002b:
- Se añade `log_freq` (empírica intramuestral, idéntica metodología que exp_002b) y se reporta también `ρ_parcial(surprisal, drift_l | log_freq)` por capa. No se modifica el conjunto de tokens.

## 4. Configuración

Ver `config.yaml`.

Decisiones clave:
- Modelo, dataset, filtros, seed y skip de warmup: **idénticos a exp_001/exp_002**. Los `n_tokens` resultantes deben ser 9 955 y la columna `surprisal_bits` debe coincidir token-a-token con exp_002 (verificación post-ejecución; condición necesaria de coherencia).
- Capas barridas: `[0, 4, 8, 12, 16, 20, 23, -1]`. Cobertura del rango con paso ≈ 4 y dos puntos en la zona terminal (23 = pre-final-LN; -1 = post-final-LN) para diagnosticar si el efecto se concentra en el LayerNorm final.
- Bootstrap: B = 1000, α = 0.05.
- Parcial: `pingouin.partial_corr(method="spearman")`, mismo método que exp_002b.

**Implementación — coste computacional y elección de estrategia.**

Dos opciones técnicas para extraer M2 en N capas:

| Opción | Forwards por documento | Memoria temporal por doc | Tiempo extra vs exp_002 |
|---|---|---|---|
| A. Un forward, extraer N capas (escogida) | 1 | ~26 MB (tuple completa de hidden_states; Pythia 1.4B = 25 × (1×256×2048) bf16) | ≈ 0 % en GPU/MPS (norm y resta vectorizadas); ≈ 5-10 % en host por la lectura |
| B. N forwards independientes, uno por capa | N = 8 | igual (output_hidden_states es necesario en todas) | ~700 % (8× la GPU) |

Se elige **A**: `output_hidden_states=True` ya devuelve la tupla completa, así que un único forward proporciona todas las capas y la opción B sería computacionalmente absurda. Se implementa vía `iat.metrics.activation_drift_multilayer(lm, ids, layer_indices)` (un único forward, extrae las N capas pedidas) — añadido a `src/iat/metrics/activation.py` junto a la versión extendida de `activation_drift_last_layer(lm, ids, layer_idx=-1)`. El test de equivalencia (`test_activation_drift_multilayer_matches_single`) verifica que ambas rutas dan resultados idénticos por capa.

Memoria pico estimada: ~26 MB / doc por encima del baseline de exp_002. M3 Pro 16 GB tiene holgura amplia.

## 5. Ejecución

```bash
python experiments/exp_003_drift_by_layer/run.py
```

Tiempo estimado en M3 Pro: ~1 min de cómputo puro (mismo orden que exp_002, single forward por doc) + tiempo de carga de modelo (pesos ya en cache local tras exp_002, ~2 s) + ~10 s de bootstrap + ~5 s de parciales pingouin = **≈ 1-2 min wall**. Si el modelo no está en caché, sumar 1-2 min de descarga (290 shards).

Si el run satura memoria en MPS (no esperado, < 30 MB extra por doc), reducir `max_docs` aquí; **no tocar exp_001 ni exp_002**.

**Pendiente de ejecución por instrucción explícita: este experimento queda pre-registrado en commit aparte y se ejecuta en sesión posterior tras revisión del diseño.**

## 6. Resultados

Run de referencia: `results/20260513T201230Z/`.

Coherencia con exp_002 verificada: en `l = -1` el ρ marginal observado es **−0.0436**, exactamente el valor reportado por exp_002 y exp_002b (los tres experimentos comparten los mismos 9 955 tokens y la columna `surprisal_bits` es idéntica byte a byte). Esto descarta cualquier deriva involuntaria en la cadena de cómputo o en la métrica extendida.

| capa l | drift media | drift mediana | ρ marginal | IC 95% bootstrap | p (marginal) | ρ parcial \| log_freq | IC 95% Fisher | p (parcial) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (embeddings) | 0.028 | 0.029 | +0.0150 | [−0.004, +0.035] | 0.134 | −0.0950 | [−0.11, −0.08] | 2.1 × 10⁻²¹ |
| 4 | 0.936 | 0.756 | −0.0061 | [−0.026, +0.015] | 0.544 | −0.1176 | [−0.14, −0.10] | 5.7 × 10⁻³² |
| 8 | 1.156 | 0.938 | **+0.0477** | [+0.028, +0.069] | 1.9 × 10⁻⁶ | **−0.0403** | [−0.06, −0.02] | 5.7 × 10⁻⁵ |
| 12 | 1.240 | 1.018 | −0.0894 | [−0.110, −0.068] | 4.0 × 10⁻¹⁹ | −0.1409 | [−0.16, −0.12] | 2.5 × 10⁻⁴⁵ |
| 16 | 1.554 | n/d | **−0.1562** | [−0.176, −0.135] | 1.9 × 10⁻⁵⁵ | **−0.1895** | [−0.21, −0.17] | 4.6 × 10⁻⁸¹ |
| 20 | 2.068 | n/d | −0.1178 | [−0.139, −0.097] | 8.1 × 10⁻³² | −0.1533 | [−0.17, −0.13] | 4.5 × 10⁻⁵³ |
| 23 (pre-final-LN) | 2.322 | n/d | −0.0784 | [−0.099, −0.057] | 4.4 × 10⁻¹⁵ | −0.1325 | [−0.15, −0.11] | 1.7 × 10⁻⁴⁰ |
| −1 (post-final-LN) | 2.124 | n/d | **−0.0436** | [−0.064, −0.024] | 1.4 × 10⁻⁵ | **−0.1429** | [−0.16, −0.12] | 1.4 × 10⁻⁴⁶ |

Argmax sobre ρ con signo:
- **max ρ_marginal = +0.0477** en l = 8.
- **max ρ_parcial = −0.0403** en l = 8 (es decir, el "menos negativo"; sigue siendo negativo en valor absoluto).

Argmax sobre |ρ|:
- max |ρ_marginal| = 0.1562 en l = 16 (signo negativo).
- max |ρ_parcial| = 0.1895 en l = 16 (signo negativo).

**La curva ρ(l) no tiene un máximo positivo interior.** Tiene en cambio un **mínimo interior** en l = 16, donde la asociación negativa es más fuerte, atenuándose después hacia l = 20, 23 y especialmente l = -1 (post-final-LN). El embedding (l = 0) muestra ρ ≈ 0 (drift mean = 0.028, prácticamente plano por construcción de la entrada).

Artefactos del run: `config.snapshot.yaml`, `summary.json` (per-layer + max), `records.parquet` (~707 KB; 11 columnas, 9 955 filas: doc_idx, position, token_id, surprisal_bits, 8× drift_l*, log_freq), `git_sha.txt`, `env.txt`.

## 7. Análisis estadístico

**Comparación con el pre-registro (sección 2).**

| Predicción | Umbral | Observado | ¿Se cumple? |
|---|---|---|---|
| Principal: `max_l ρ_marginal(surprisal, drift_l) >= 0.25` | 0.25 | +0.0477 (l = 8) | **No** |
| IC inferior del argmax > 0.10 | 0.10 | IC inferior +0.028 | **No** (ni siquiera el IC inferior alcanza 0.10) |
| De forma: argmax interior (l ∈ {8, 12, 16}) | l ∈ {8, 12, 16} | argmax ρ con signo en l = 8 | Cumple coordenada pero el valor en l = 8 es ~0.05, irrelevante |
| Complementaria: `ρ_parcial(argmax) >= 0.20` | 0.20 | −0.0403 (l = 8) | **No** |
| Falsación absoluta: `max_l ρ_marginal < 0.25` Y `max_l ρ_parcial < 0.20` | ambas | +0.0477 y −0.0403 | **Sí — falsación pre-registrada se dispara** |

La predicción cuantitativa principal **se falsifica con margen amplio**. Ninguna de las 8 capas — incluyendo los puntos diagnósticos l = 23 (pre-final-LN) y l = -1 (post-final-LN) que servían para diseccionar el efecto del LayerNorm final — produce una correlación positiva con surprisal mayor a 0.048. Ni siquiera reformulando la predicción sobre |ρ| (variante post-hoc, marcada como tal) supera el umbral: max |ρ_marginal| = 0.156.

**Estructura por capa, lectura cualitativa (post-hoc, marcada como tal).**

- `drift_mean(l)` crece monótonamente con l hasta l = 23 (de 0.028 a 2.322) y baja ligeramente en l = -1 (2.124) por la normalización final. Coherente con que las capas profundas acumulan transformación.
- ρ_marginal(l) pasa de ~0 en el embedding a positiva pequeña en l = 8, cruza cero, y se hace **cada vez más negativa hasta un mínimo en l = 16 (ρ = −0.156)**, después se atenúa hacia capas más profundas y hacia el post-norm.
- ρ_parcial(l | log_freq) es negativa en **todas** las capas barridas (rango [−0.190, −0.040]) y significativa con p ≤ 5.7 × 10⁻⁵ en todos los casos. Es decir: a frecuencia léxica fija, surprisal y drift se asocian negativamente independientemente de la capa.
- El efecto principal de `log_freq` sobre drift, ya cuantificado en exp_002b (β_OLS = −0.098 en la última capa), parece ser dominante también en capas intermedias: la inversión de signo entre ρ_marginal positivo (l = 8) y ρ_parcial negativo (l = 8) replica el mismo patrón que en exp_002b para l = −1.
- La capa l = 0 (embeddings de entrada) tiene drift media 0.028, dos órdenes de magnitud menor que las demás. Es consistente con que `hidden_states[0]` no haya pasado por ningún bloque transformer y refleje sólo la diferencia entre embeddings consecutivos en una pequeña ventana. No es informativo para H1.

**Sanity numérico explícito.**
- `ρ_marginal(l = −1) = −0.0436` coincide token-a-token con exp_002 (sección 6 de su README) y con exp_002b (sección 6, métrica marginal). La consistencia de cadena de cómputo está verificada.
- `ρ_parcial(l = −1 | log_freq) = −0.1429` coincide token-a-token con exp_002b sección 6. Verificado.

## 8. Conclusiones

**La predicción principal queda falsificada. El criterio de falsación absoluto pre-registrado en sección 2 ("max_l ρ_marginal < 0.25 Y max_l ρ_parcial < 0.20") se dispara con holgura.** Ninguna capa barrida produce la asociación positiva monótona predicha entre surprisal Shannon y norma del cambio de hidden state.

Lectura honesta:

1. **La línea de drift geométrico (M2) queda formalmente cerrada como métrica primaria de información**, no sólo en la última capa (ya falsificada en exp_002) sino en todo el rango de capas que un argumento de "saltar el LayerNorm final" podría rescatar (l = 0, 4, 8, 12, 16, 20, 23). El punto diagnóstico crítico — l = 23, pre-final-LN — produce ρ_marginal = −0.078 y ρ_parcial = −0.133, indistinguible de l = -1 en términos de magnitud y signo. La conjetura de exp_002b sección 8 ("capas previas no normalizadas deberían comportarse distinto") **no se sostiene empíricamente**. La normalización final no era la causa.
2. La señal estadísticamente robusta que sí aparece es **negativa, pequeña, y pico en capas medias-tardías** (l = 16 alcanza |ρ_parcial| = 0.190). Es decir, en Pythia 1.4B, a frecuencia léxica fija, tokens más sorpresivos se asocian con drift **menor** (no mayor) en hidden states, máximo de ese efecto entre capas 12 y 16. Esto es contra-intuitivo respecto al marco "información como transformación" original y merece interpretación, pero no rescata la métrica como predictor de información en el sentido en que se pre-registró.
3. La consistencia byte-a-byte con exp_002 en l = −1 elimina cualquier sospecha de bug en la métrica extendida `activation_drift_last_layer(..., layer_idx)` o en `activation_drift_multilayer`. Los tests parametrizados (8 en total, ver `tests/test_smoke.py`) ya verifican equivalencia single ↔ multi-pass; los datos lo confirman in vivo.
4. El argumento alternativo "tal vez la norma L2 no es la métrica correcta para el drift, probar cosine o effective dimension" (M4, M5 del diseño original) **podría ser una vía**, pero sin justificación teórica adicional sería **picking statistics post-hoc** sobre una hipótesis ya falsificada con dos pre-registros consecutivos. Se documenta esta opción en la ADR 0004 (versión B) y se deja como no recomendada.
5. Lo único que queda firmemente establecido empíricamente es: en Pythia 1.4B sobre WikiText, la cantidad geométrica `||h_l(t_{1..i}) − h_l(t_{1..i−1})|| / √d_l`, en cualquiera de las 8 capas barridas, **no codifica la sorpresa Shannon en sentido monótono positivo**. La intuición motivacional ("tokens sorpresivos cambian más el estado interno") no se realiza en esta especificación.

## 9. Próximos pasos sugeridos

Decisión: pivotar a **M1 (Bayesian surprise sobre bloque futuro)** como nueva métrica primaria candidata, abandonando M2 geométrico. Justificación, contexto histórico y consecuencias en `docs/decisions/0004-cierre-linea-drift-geometrico.md`.

Próximo experimento concreto:

- **exp_004 — M1 Bayesian surprise sobre bloque futuro k = 5**, sobre los mismos 9 955 tokens. M1 es la KL entre la distribución predictiva sobre los siguientes k tokens *antes* y *después* de observar t_i. Pre-registro a redactar en el README de exp_004 sin ejecutar:
  - Predicción 1 (sanity): ρ_Spearman(M1, surprisal) ∈ [0.4, 0.8]. M1 hereda estructura probabilística de Shannon, deben correlacionar más que drift (que estaba alrededor de 0).
  - Predicción 2 (la verdaderamente relevante): ρ_parcial(M1, T1_cloze | surprisal) > 0.15. Es la pregunta de fondo del plan: ¿hay *algo* ortogonal a Shannon con poder predictivo sobre tareas downstream?
  - Diseño: requiere implementar T1_cloze (ver `docs/design.md` §4 T1) y un dataset con probes funcionales. Documentar en el README de exp_004 si se usa LAMBADA, Counterfact, o un cloze sintético inicial.

Se descarta explícitamente:
- **exp_002c** (réplica de exp_002b con frecuencias de corpus externo): el resultado de exp_003 hace que el confound de frecuencia ya no sea la incógnita pendiente. exp_002c se queda como control de robustez de baja prioridad, **no se ejecuta** salvo que haya una razón teórica nueva.
- **Variantes de M2** (cosine drift M4, effective dim M5): sólo se considerarán si exp_004 también falla y queremos hacer un barrido sistemático de geometría con nueva motivación teórica. Hasta entonces, no.
