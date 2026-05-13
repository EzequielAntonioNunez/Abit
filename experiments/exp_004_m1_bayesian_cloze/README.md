# Proyecto de investigación de Abit (Identy Labs)

# exp_004 — Bayesian surprise (M1, k=5) y delta cloze intra-documento

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: 2026-05-13 (22:30:31 → 22:55:21 hora local, UTC 20:30:31Z, wall ~24 min 50 s)
- **Autor**: Ezequiel
- **Claude Code model**: claude-opus-4-7 (1M context)
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

Run de referencia: `results/20260513T203031Z/`.

Coherencia con experimentos previos: los `surprisal_bits` calculados desde `m1_and_cloze_block` no son token-a-token comparables a exp_001 porque exp_004 reduce el rango de targets a `[skip, L − 1 − k] = [50, 250]` (en exp_001-003 era `[skip, L − 1] = [50, 255]`). N de exp_004 = 9 563 vs N de exp_001/002/003 = 9 955. Los descriptivos de surprisal son coherentes en mean/median/p95 (mean 3.629 vs 3.639 en exp_001) con la pequeña diferencia esperada por los 392 tokens descartados al final de cada doc.

| Métrica | Valor |
|---|---|
| Modelo / device / dtype | `EleutherAI/pythia-1.4b` / `mps` / `torch.bfloat16` |
| k (tamaño bloque futuro) | 5 |
| Documentos usados | 98 (idénticos a exp_001-003 en el filtro min_chars) |
| Tokens evaluados (N) | 9 563 |
| surprisal_bits | mean 3.629 / median 2.298 / p95 11.587 |
| m1_kl_block_nats | mean 9.071 / median 8.204 / p95 19.149 |
| delta_cloze | mean 0.288 / median 0.097 / p95 1.191 |
| Tiempo real de ejecución | ~24 min 50 s (≈ 9 563 forwards B + carga inicial) |
| SHA git en el run | `9b454c2` |

**Correlaciones por predicción** (Spearman; IC 95% bootstrap para marginales con B = 1000, IC 95% Fisher de pingouin para parciales):

| Predicción | Tipo | Métrica | Observado | IC 95% | p-valor | Predicho | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| **P2 (PRIMARIA)** | parcial | ρ(M1, delta_cloze \| surprisal) | **+0.1097** | [+0.09, +0.13] | 5.7 × 10⁻²⁷ | > 0.15 | **No** (por 0.04) |
| P1 (sanity M1 ↔ Shannon) | marginal | ρ(M1, surprisal_bits) | **−0.0529** | [−0.074, −0.031] | 2.2 × 10⁻⁷ | ∈ [0.4, 0.8] | **No** (signo invertido) |
| P3 (sanity cloze ≠ ruido) | marginal | ρ(delta_cloze, surprisal_bits) | +0.1442 | [+0.124, +0.163] | 1.3 × 10⁻⁴⁵ | IC excluye 0 | **Sí** |
| P4 (M1 vs log_freq confound) | parcial | ρ(M1, surprisal_bits \| log_freq) | −0.1318 | [−0.15, −0.11] | 2.7 × 10⁻³⁸ | > 0.3 | **No** (signo invertido) |

Artefactos: `config.snapshot.yaml`, `summary.json` (descriptivos + las 5 correlaciones), `records.parquet` (~282 KB; columnas `doc_idx, position, token_id, surprisal_bits, m1_kl_block_nats, delta_cloze, log_freq`), `git_sha.txt`, `env.txt`.

## 7. Análisis estadístico

**Lectura jerarquizada según pre-registro (sección 2):**

1. **P1 falla rotundamente**: ρ(M1, Shannon) = −0.0529 con IC 95% [−0.074, −0.031], **estadísticamente significativo del lado opuesto al esperado**. La predicción era ρ ∈ [0.4, 0.8]. La distancia entre el observado y el límite inferior del rango pre-registrado es ~0.45, un orden de magnitud mayor que el ancho del IC. No es ruido: M1 y Shannon están casi descorrelacionadas y con tendencia ligeramente *opuesta*.
2. **P4 falla con el mismo patrón**: ρ_parcial(M1, Shannon | log_freq) = −0.1318. Si M1 fuese una métrica de información Shannon-like (incluso ajustada por log_freq), esto debería ser > 0.3. No hay nada que rescate aquí controlando frecuencia.
3. **P3 pasa**: ρ(delta_cloze, surprisal) = +0.1442, IC estrictamente positivo. La señal cloze NO es ruido — hay asociación monótona positiva entre lo sorpresivo de t_i y el cambio en log-prob del token a 5 pasos vista. Esto es relevante para exp_004b: la rama T1 funciona, el aparato cloze produce señal medible.
4. **P2 (primaria) no se puede interpretar limpiamente**: ρ_parcial(M1, delta_cloze | Shannon) = +0.1097, IC [+0.09, +0.13]. **Falla la predicción cuantitativa** (> 0.15) pero el resultado queda en zona ambigua. Sin embargo, dado que P1 falla, **no se puede atribuir este parcial a "M1 captura algo que Shannon no": la M1 implementada simplemente no es la métrica que pretendía ser** (ver sección 8). El parcial es el de la métrica que hay, no el de la métrica diseñada.

**El veredicto del pre-registro es claro**: bajo la jerarquía de la sección 2, P1 es prerrequisito y P1 falla. Por tanto P2 no se interpreta como respuesta a la pregunta de investigación. **Estamos en CASE C — fallo de instrumentación, no resultado teórico negativo.**

## 8. Conclusiones

Esta sección documenta el bug honestamente, distingue entre fallo conceptual y fallo numérico, y NO interpreta el resultado como negativo de la teoría general — porque no podemos decir que la teoría falla si la métrica que pretendíamos usar no es la métrica que la teoría requiere.

### Diagnóstico — bug conceptual primario

La función `iat.metrics.bayesian.m1_and_cloze_block`, tal como está implementada en el commit ejecutado (`9b454c2`), calcula:

```
M1(i) = Σ_{j=1..k} KL( softmax(logits_A[i+j-1]) || softmax(logits_B[i+j-2]) )
```

Para `j = 1`, esto es:

```
KL( p(• | t_0..t_i) || p(• | t_0..t_{i-1}) )
```

El problema no es la matemática del KL, que está bien calculado por el código. El problema es **qué representan estas dos distribuciones**:

- `p(• | t_0..t_i)` es la distribución del modelo sobre la **posición i + 1** (es decir, predice el siguiente token tras consumir hasta t_i).
- `p(• | t_0..t_{i-1})` es la distribución sobre la **posición i** (predice t_i, o lo que el modelo cree que viene tras t_{i−1}).

Son distribuciones sobre **distintas posiciones de la secuencia**, no sobre la misma variable aleatoria. El KL entre ellas no mide "información que t_i aporta sobre el bloque futuro" — la definición de Bayesian surprise de Itti & Baldi y de `docs/design.md` §3 M1. Lo que mide es "cuánto cambia el output del autoregressive head al avanzar un paso del cursor", que para LMs en texto natural está **dominado por la forma de p_before**: cuando t_i es predecible, `p(• | t_0..t_{i-1})` está fuertemente concentrada sobre t_i y muy poco peso queda fuera del peak — log-probabilidades muy negativas en el complemento — lo que hace que el sumando `(log p_after − log p_before)` sea grande sobre los modos de p_after. Resultado: M1 grande para tokens predecibles, M1 pequeño para tokens sorpresivos. Negativamente correlacionado con surprisal por construcción. Eso es exactamente el patrón empírico (ρ = −0.053).

La definición correcta de M1 (KL de bloque futuro condicional vs no condicional sobre t_i) requiere la distribución *marginal* `p(t_{i+1} | C) = Σ_v p(v | C) · p(t_{i+1} | C, v)`. Esta marginalización es lo que mi implementación intentaba aproximar con la "secuencia con t_i removido" — pero la aproximación por skip-and-shift en un AR LM no implementa la marginal: implementa **otra cosa** (predicción de "siguiente tras t_{i−1}", que el modelo entiende como predicción de t_i, no de t_{i+1}). El skip-and-shift en AR causal mask no es equivalente a marginalizar; cualquier persona que escribió el código (yo) debería haberlo visto en revisión, y no lo hice. Lo asumí.

### Issue numérico secundario (efecto pequeño pero presente)

Sanity adicional ejecutado en sesión: para una secuencia controlada de longitud 10, las diferencias en logits entre forward A y forward B en posiciones que el causal mask predice idénticas son:

- **bfloat16 / MPS**: hasta ~0.17 en magnitud absoluta de logit, posición a posición.
- **float32 / CPU**: hasta ~1.0 × 10⁻⁴ (ruido numérico esperado).

El ruido bfloat16/MPS NO es la causa principal del nulo (eso es el bug conceptual), pero contribuye degradando aún más cualquier señal residual al exponenciar logits con error 0.17 y derivar log-probabilidades para la KL.

### Lo que NO se puede concluir desde este experimento

No se puede concluir:
- Que la formulación probabilística de "información como transformación" falla en este setup. La métrica ejecutada no es esa formulación.
- Que M1 (correctamente definida) sea redundante con Shannon. No se ha medido M1 correctamente.
- Que delta_cloze sea inútil como señal funcional. P3 mostró que sí tiene señal positiva con Shannon (ρ = +0.144), y el parcial de la M1-bug contra delta_cloze (ρ_parc = +0.110) sugiere que **alguna** señal funcional existe incluso a través de una métrica defectuosa, lo cual deja la pregunta de fondo abierta.

### Lo que sí se puede afirmar

- El aparato cloze (forward A + B + lectura de log-prob de t_{i+k}) **funciona y produce señal medible** (P3 pasa). Esa rama del pipeline es reutilizable.
- La métrica M1 actual debe reemplazarse antes de cualquier interpretación. exp_004 queda como histórico inmutable de un fallo de instrumentación pre-registrado.

## 9. Próximos pasos sugeridos

**Propuesta de exp_004b — re-ejecución tras fix de M1.**

No se crea aún la carpeta del experimento; se deja como propuesta de diseño para revisión humana antes de proceder.

Tres opciones de reformulación, ordenadas por preferencia:

1. **M1 vía argmax-counterfactual** (preferida por coste/claridad).
   ```
   M1_cf(i) = log p(t_{i+1..i+k} | C, t_i) - log p(t_{i+1..i+k} | C, v_top)
   ```
   donde `v_top = argmax_v p(v | C)` es lo que el modelo *esperaba* en la posición i. Las dos ramas evalúan la log-probabilidad del bloque continuación real bajo dos posibles "tokens en posición i": el real (t_i) y el más probable según el modelo (v_top).
   - Si t_i = v_top (predecible): M1_cf ≈ 0.
   - Si t_i ≠ v_top (sorpresivo): M1_cf puede ser grande positivo o negativo según si t_i ayuda o estorba a predecir el bloque real.
   - Coste: 1 forward A por documento + 1 forward por target token con `[..., v_top, ...]` en lugar de `[..., t_i, ...]`. Mismo orden de coste que exp_004.
   - Predicción esperada: ρ(M1_cf, surprisal) en [0.3, 0.7] (positiva, porque tokens sorpresivos producen mayor divergencia en log-prob del bloque real cuando t_i se reemplaza por v_top).

2. **M1 vía marginalización top-K**: aproximar `p(t_{i+1} | C) ≈ Σ_{v ∈ top-K} p(v|C) · p(t_{i+1} | C, v)` con K ∈ {5, 10}. Más fiel a la definición original pero K × más caro: K = 10 implica ~95 000 forwards (~4 h en M3 Pro). Considerar sólo si (1) deja resultado ambiguo.

3. **M1 vía forward A únicamente, redefinida como sensibilidad**: M1_sens(i) = KL( p(• | t_0..t_i) || p(• | t_0..t_{i-1}) ) sin segundo forward, asumiendo (por causal mask exacto) que ambas predicciones vienen del mismo forward A. Esto es matemáticamente idéntico a lo que computé en exp_004 (módulo el ruido bfloat16), pero **NO arregla el bug conceptual** — sería renombrar el problema. Mencionada aquí para descartarla explícitamente: la sensibilidad consecutiva es lo que ya medimos, y ya sabemos que correlaciona con −Shannon.

**Antes de implementar exp_004b**, decisiones que necesita el humano:

- Confirmar la opción 1 (counterfactual con v_top) como reformulación canónica de M1.
- Confirmar que delta_cloze se mantiene como T1 (P3 valida que produce señal).
- Decidir si se ejecuta en bfloat16/MPS o se pasa a float32/CPU para la rama del segundo forward (CPU es mucho más lento pero elimina el ruido secundario).

exp_005 **no se planifica**. La línea queda en pausa hasta tener una M1 que pase su propio sanity check antes de poder contestar la pregunta de fondo.
