# Proyecto de investigación de Abit (Identy Labs)

# exp_003 — Activation drift, barrido por capa

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: `[pendiente — diseño pre-registrado, ejecución posterior]`
- **Autor**: Ezequiel
- **Claude Code model**: `[rellenar tras ejecutar]`
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

`[rellenar tras ejecutar]`

Última ejecución: `[results/<timestamp>/summary.json]`

## 7. Análisis estadístico

`[rellenar tras ejecutar]`

## 8. Conclusiones

`[rellenar tras ejecutar]`

## 9. Próximos pasos sugeridos

`[rellenar tras ejecutar]`
