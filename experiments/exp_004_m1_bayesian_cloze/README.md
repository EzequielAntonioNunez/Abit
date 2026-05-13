# Proyecto de investigación de Abit (Identy Labs)

# exp_004 — Bayesian surprise (M1, k=5) y delta cloze intra-documento

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: `[pendiente — diseño pre-registrado, ejecución posterior tras revisión]`
- **Autor**: Ezequiel
- **Claude Code model**: `[rellenar tras ejecutar]`
- **Hardware**: M3 Pro local, backend `mps`, dtype `bfloat16`

## 1. Hipótesis

Tras el cierre formal de la línea de drift geométrico en `docs/decisions/0004-cierre-linea-drift-geometrico.md`, la métrica primaria candidata pasa a ser **M1 — Bayesian surprise sobre bloque futuro** (`docs/design.md` §3). exp_004 evalúa simultáneamente dos preguntas distintas, ambas necesarias para que la línea de "información ortogonal a Shannon" siga viva:

1. **Sanity probabilístico (predicción de coherencia, no de descubrimiento)**: M1 hereda estructura probabilística de Shannon — ambos viven en el mismo espacio de log-probabilidades del modelo —, por lo que deben correlacionar fuertemente. Es lo opuesto al fallo de drift, donde la métrica vivía en un espacio geométrico ortogonal al de Shannon y por eso podía no correlacionar; aquí, si M1 no correlaciona con Shannon, es señal de bug.

2. **Pregunta de fondo del plan (predicción de descubrimiento)**: ¿hay *algo* en M1 que no esté ya en Shannon y que tenga poder predictivo sobre una tarea funcional? Operacionalizamos esto con la primera versión de **T1 cloze** (`docs/design.md` §4): el cambio de log-probabilidad del token a `k` pasos vista cuando observamos t_i (`delta_cloze`). Si M1 aporta señal ortogonal a Shannon sobre delta_cloze, hay base empírica para H3 del plan general.

## 2. Predicción pre-registrada

Las predicciones se separan deliberadamente en dos niveles: **una predicción PRIMARIA** (la única que cuenta como pregunta de investigación) y **tres sanity checks** (que sólo validan que el aparato está funcionando; pasarlas no es un resultado, fallarlas obliga a depurar antes de interpretar).

### Predicción PRIMARIA (resultado)

| # | Predicción | Métrica | Umbral | Lectura |
|---|---|---|---|---|
| **P2** | M1 aporta señal predictiva ortogonal a Shannon sobre una tarea funcional | Spearman parcial ρ(M1, delta_cloze \| surprisal_bits) | **> 0.15** | Es la única pregunta de investigación que exp_004 puede contestar. Si se cumple, hay primer apoyo empírico positivo de H3 del plan general en este proyecto. Si no, la formulación probabilística de "información como transformación" tampoco se detecta. |

### Sanity checks (NO son resultados; sólo validan el setup)

| # | Predicción | Métrica | Umbral | Si falla, qué significa |
|---|---|---|---|---|
| P1 | M1 correlaciona con Shannon (coherencia metodológica) | Spearman ρ(M1, surprisal_bits) | ∈ [0.4, 0.8] | ρ < 0.3 → bug en `m1_and_cloze_block`. ρ > 0.85 → M1 colapsa a Shannon y P2 pierde sentido. No es un resultado; es un prerrequisito para creer en P2. |
| P3 | delta_cloze no es ruido | Spearman ρ(delta_cloze, surprisal_bits) | IC 95% excluye 0 | Si IC contiene 0, la señal cloze es indistinguible de ruido y P2 no puede contestar la pregunta porque no hay tarea funcional medible. Bug en T1 o falta de poder estadístico. |
| P4 | M1 no es subproducto de frecuencia léxica | Spearman parcial ρ(M1, surprisal_bits \| log_freq) | > 0.3 | Control análogo al de exp_002b: si M1 cae mucho tras controlar log_freq, sospecha de confound. No invalida P2 pero la matiza. |

### Criterio de falsación de la teoría general

Si **P1 pasa** (M1 está bien implementada, correlaciona como Shannon-like debería) **y P2 < 0.05** (M1 no aporta señal ortogonal sobre delta_cloze por encima del ruido), entonces:

> **La formulación probabilística de "información como transformación" tampoco se detecta en este setup.**

Esto sería el cuarto pre-registro consecutivo del proyecto que falla (exp_002 M2 última capa; exp_003 M2 capas; ahora M1 ortogonalidad). En ese caso **no se abre exp_005 automáticamente**: hay que revisar el marco teórico antes de seguir corriendo experimentos. La decisión queda en revisión humana, no en planificación automática.

Si P2 ∈ [0.05, 0.15) — zona ambigua entre "no se detecta" y "se detecta" — el resultado se documenta como inconcluso. Tampoco se abre exp_005 sin decisión humana.

### Notas sobre dependencias entre predicciones

- **P1 es prerrequisito metodológico**, no resultado. Pasar P1 no aporta evidencia sobre la teoría.
- **P2 es la única predicción que cuenta como resultado de investigación.** Falle como falle, lo que diga P2 (dentro o fuera del umbral) es lo que se reporta como hallazgo del experimento.
- **P3 y P4 son controles**. Si P3 falla, P2 no puede contestar la pregunta (el aparato cloze está roto). Si P4 falla pero P2 pasa, el resultado es real pero matizado por el confound de frecuencia y exp_005 debería re-validarlo con frecuencias externas.

## 3. Cambios respecto al experimento anterior

Cambio principal único respecto a exp_003: se introduce la métrica **M1 — Bayesian surprise sobre bloque futuro k = 5** (`docs/design.md` §3) y la primera operacionalización funcional **T1 — delta_cloze intra-documento a offset k** (`docs/design.md` §4), abandonando M2 según ADR 0004.

Cambios menores derivados:
- Por el coste de M1 (dos forwards por target token), se redefine el rango de targets útiles a `[skip, L - 1 - k]` por documento — necesario para que t_{i+k} exista.
- Se reporta correlación parcial controlando por surprisal (no por log_freq) como métrica primaria de P2; la parcial controlando por log_freq se reporta como control adicional.

## 4. Configuración

Ver `config.yaml`.

Decisiones clave:
- Modelo, dataset, filtros, seed y skip de warmup: **idénticos a exp_001/exp_002/exp_003**. Los `surprisal_bits` calculados desde el forward A en `m1_and_cloze_block` deben coincidir token-a-token con `shannon_surprisal` por construcción (test `test_m1_surprisal_matches_shannon` en `tests/test_smoke.py` lo verifica). Esto da una alineación trivial y un sanity numérico inmediato.
- `k = 5` (tamaño del bloque futuro). Cumple `k ∈ {1, 5, 20}` del diseño original (`docs/design.md` §3) y es el valor pre-registrado en este experimento.
- `delta_cloze` se evalúa al offset k_probe = k = 5. Esto permite reutilizar el mismo forward B por target tanto para M1 como para el probe cloze (eficiencia 2x sobre cómputo independiente).
- **Frecuencia léxica** para los parciales de control: empírica intramuestral, idéntica metodología que exp_002b/exp_003 (con la misma limitación documentada de singletons → log_freq = 0).
- Bootstrap percentil B = 1000, α = 0.05 para correlaciones marginales; IC Fisher de pingouin para parciales.

**Estructura computacional del experimento**:

| Componente | Forwards por documento | Coste estimado por doc (Pythia 1.4B, M3 Pro) |
|---|---|---|
| Forward A (compartido) | 1 | ~25-50 ms |
| Forwards B (uno por target) | ~200 targets | ~5-10 s |
| KL block por target (sum_{j=1..k} sobre V=50304) | ~200 × k = 1000 sumas de vocab | <1 s |
| delta_cloze por target (1 indexación por target) | ~200 indexaciones escalares | trivial |

Coste total estimado: **~10-20 min wall** en M3 Pro para 100 docs (≈ 20 000 forwards B + 100 forwards A + 100 forwards de carga compartidos con shannon en exp_001 si los reusáramos, que no es el caso aquí). Si saturamos memoria de MPS, se reduce `max_docs` aquí — **no en exp_001/002/003**.

**Sobre datasets externos (Counterfact, LAMBADA)**: el config los lista en `cloze.external_datasets.candidates` con `enabled: false`. exp_004 deliberadamente usa probes intra-documento como primera validación; introducir datasets externos al mismo tiempo que la métrica nueva mezcla dos cambios. Si exp_004 muestra señal positiva en P2, exp_005 introducirá Counterfact y/o LAMBADA con su propio pre-registro. Si exp_004 falla en P2, los datasets externos podrían rescatarlo (mayor poder estadístico, probes más limpios) pero la decisión la toma el humano tras ver exp_004.

## 5. Ejecución

```bash
make exp_004
```

Tiempo estimado: 10-20 min wall en M3 Pro (mayoritariamente forwards B sobre ~20 000 secuencias de longitud 255).

**Pendiente de ejecución por instrucción explícita: este experimento queda pre-registrado en commit aparte y se ejecuta en sesión posterior tras revisión del diseño y autorización.**

## 6. Resultados

`[rellenar tras ejecutar]`

Última ejecución: `[results/<timestamp>/summary.json]`

## 7. Análisis estadístico

`[rellenar tras ejecutar]`

## 8. Conclusiones

`[rellenar tras ejecutar]`

## 9. Próximos pasos sugeridos

`[rellenar tras ejecutar]`
