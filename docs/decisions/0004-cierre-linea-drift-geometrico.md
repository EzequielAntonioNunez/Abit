# ADR 0004 — Cierre de la línea de drift geométrico como métrica primaria

**Fecha**: 2026-05-13
**Estado**: aceptada

## Contexto

La métrica M2 — `activation drift` (norma L2 normalizada del cambio entre estados ocultos consecutivos por capa) — se introdujo en `docs/design.md` §3 como una de las cinco candidatas a capturar "información ortogonal a Shannon" (hipótesis H3 del plan, `docs/plan.md` §2). Tres experimentos consecutivos han evaluado esta línea con pre-registros independientes y todos han fallado:

- **exp_001** validó el pipeline: surprisal Shannon en Pythia 1.4B sobre WikiText-103 produce perplejidad 12.46 y distribución coherente con la literatura. Sin anomalías de implementación.
- **exp_002** evaluó M2 en la última capa (`hidden_states[-1]`, post-final-LN). Predicción pre-registrada: Spearman ρ(surprisal, drift_last) ∈ [0.3, 0.6]. Observado: ρ = −0.044, IC 95% [−0.064, −0.024]. **Falsada.** El análisis exploratorio post-hoc reveló una U-shape no monótona por quintiles de surprisal.
- **exp_002b** descompuso el patrón controlando por log-frecuencia léxica. Tres hipótesis pre-registradas: H1 (artefacto de LayerNorm), H2 (compresión semántica), H3 (confound de frecuencia). Observado: ρ_parcial = −0.143, IC 95% [−0.16, −0.12], U-shape persiste en los 9 estratos de log_freq. H2 y H3 descartadas con claridad; H1 mejor soportada cualitativamente aunque el partial excede ligeramente su rango cuantitativo [−0.1, 0.1].
- **exp_003** barrió M2 en 8 capas, `l ∈ {0, 4, 8, 12, 16, 20, 23, −1}`, con la predicción pre-registrada de que al menos una capa l\* alcanzaría ρ ≥ 0.25 (rescate de M2 fuera del post-final-LN). Observado: max_l ρ_marginal = +0.048 (l = 8), max_l ρ_parcial = −0.040 (l = 8). Criterio de falsación absoluto del pre-registro (`max_l ρ < 0.25 y max_l ρ_parcial < 0.20`) **se dispara con holgura**. Notablemente: el punto diagnóstico l = 23 (pre-final-LN) produce ρ_marginal = −0.078 y ρ_parcial = −0.133, indistinguible de l = −1 en términos de magnitud y signo, lo que **descarta** la conjetura intermedia de que la normalización final fuera la causa específica del nulo en exp_002.

La consistencia byte-a-byte en los puntos de comparación (l = −1 en exp_003 coincide con exp_002 y exp_002b hasta 4 decimales) elimina cualquier sospecha de bug en la implementación.

## Decisión

Se **cierra formalmente la línea de drift geométrico como métrica primaria de información ortogonal a Shannon** en el contexto del proyecto. Concretamente:

1. M2 (en cualquier capa, en su definición de norma L2 normalizada) deja de considerarse candidata para Fases 1-3.
2. La métrica primaria candidata pasa a ser **M1 — Bayesian surprise sobre bloque futuro** (`docs/design.md` §3), con un experimento dedicado (exp_004) que combina M1 con la primera tarea downstream T1 (cloze) para medir directamente la pregunta de fondo del plan: ¿existe alguna métrica ortogonal a Shannon con poder predictivo funcional?
3. Las variantes M4 (cosine drift) y M5 (effective dimension drift) del diseño original se **descartan** salvo nueva justificación teórica. Probarlas tras dos pre-registros falsificados sería selección post-hoc.

## Consecuencias

- exp_001, exp_002, exp_002b y exp_003 quedan como históricos inmutables con su documentación tal cual.
- `docs/plan.md` y `docs/design.md` se enmiendan con una entrada fechada que apunta a esta ADR; el cuerpo previo no se reescribe.
- El roadmap del proyecto se reorienta: Fase 1 ya no busca un barrido sistemático de M0-M5, sino que se concentra en M1 y su validación contra T1 cloze. La estructura de hipótesis general H1-H3 del plan permanece, pero su instanciación operacional cambia.
- exp_004 (M1 + T1) queda pre-registrado en commit aparte y se ejecuta tras revisión del diseño.
- La pregunta "¿hay capas previas de Pythia 1.4B donde el drift recupere señal?" queda contestada empíricamente con un no robusto. Cualquier futuro retorno a M2 deberá justificarse con teoría nueva (no con re-binning del mismo análisis) y un nuevo pre-registro.

## Alternativas consideradas

- **exp_002c** — réplica de exp_002b con frecuencias de un corpus externo (split train de WikiText-103 o C4 unigram counts) en lugar de frecuencia empírica intramuestral. **Descartada como continuación de la línea de drift**: exp_003 ya descartó que el problema fuera específico del post-final-LN, así que el confound de frecuencia (única motivación de exp_002c) deja de ser la incógnita pendiente. Si en el futuro hay una razón teórica nueva para reabrir M2, exp_002c podría rescatarse como control de robustez; hasta entonces, no se ejecuta.
- **Variantes geométricas (M4 cosine drift, M5 effective dimension)**. Descartadas por riesgo de *p-hacking* tras dos falsificaciones consecutivas con pre-registro. Si tras exp_004 (M1 + T1) ambos resultados son nulos, podría considerarse un re-análisis sistemático del espacio de métricas geométricas con nuevo pre-registro; hasta entonces, no.
- **Cambiar el modelo base a Llama 3.2 1B o Pythia 2.8B antes de cerrar M2**. Descartado por coste/riesgo: la consistencia entre las 8 capas barridas de Pythia 1.4B y el mecanismo identificado (estructura interna de la representación, no específica del LayerNorm final) hace poco probable que cambiar de modelo rescate la métrica. Replicar M2 en otra arquitectura sólo tiene sentido si M1 funciona y queremos generalizar el hallazgo.

## Refinamiento 2026-05-13

Tras revisión, el texto previo es ambiguo en un punto crítico y conviene precisarlo sin reescribir las secciones originales.

**Sobre el motivo del cierre.** El cierre de la línea M2 **no se basa en una comprensión causal positiva del nulo**. Se basa en dos hechos concretos y verificables:

1. Dos predicciones cuantitativas, pre-registradas en momentos distintos y diseñadas para discriminar entre alternativas mutuamente excluyentes, se falsificaron (exp_002: ρ ∈ [0.3, 0.6] → ρ = −0.044; exp_003: max_l ρ ≥ 0.25 → max_l ρ = +0.048).
2. No tenemos una justificación teórica nueva para una tercera iteración. Las alternativas naturales — M4 (cosine drift), M5 (effective dimension), capas no barridas, otras arquitecturas — serían re-instanciaciones del mismo marco geométrico que ya falló dos veces. Probar una tercera variante sin teoría nueva sería selección post-hoc, no investigación pre-registrada.

**Sobre la conjetura del LayerNorm.** El "Contexto" de esta ADR cita correctamente que exp_002b propuso H1 (artefacto del LayerNorm final) como mejor lectura cualitativa de la U-shape, y que exp_003 incluyó l = 23 (pre-final-LN) como punto diagnóstico. Lo que conviene explicitar es la conclusión:

- La hipótesis H1 de exp_002b **fue refutada por exp_003**. Si el LayerNorm final fuera la causa, l = 23 (pre-norm) tendría que mostrar un patrón cualitativamente distinto a l = −1 (post-norm). No lo hace: ρ_marginal(l=23) = −0.078 vs ρ_marginal(l=−1) = −0.044, y ρ_parcial(l=23) = −0.133 vs ρ_parcial(l=−1) = −0.143. Numéricamente comparables, sin discontinuidad observable.
- La frase "el mecanismo identificado (estructura interna de la representación, no específica del LayerNorm final)" en "Alternativas consideradas" debe leerse como **"el conjunto de evidencias acumuladas"**, no como una afirmación de mecanismo causal positivo. No identificamos *qué* causa el patrón observado; sólo establecimos *qué no lo causa* (no es el LayerNorm final).

**Implicación práctica.** El cierre es robusto en sus datos pero no en su interpretación causal. Cualquier reapertura futura de la línea de drift geométrico requiere — además de teoría nueva y pre-registro nuevo — una propuesta explícita de cuál sería el mecanismo subyacente, dado que la refutación de H1 (LayerNorm) deja el espacio abierto. Esto se documenta aquí para que el lector futuro no asuma que sabíamos por qué la métrica fallaba.
