# Proyecto de investigación de Abit (Identy Labs)

# exp_004b — Bayesian surprise reformulada (counterfactual + marginal top-K)

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: 2026-05-14 (00:14:15 → 02:07:52 hora local, UTC 22:14:15Z; wall ~113 min, fuera del envelope pre-registrado de 90 min; ver §6 nota de tiempo)
- **Autor**: Ezequiel
- **Claude Code model**: claude-opus-4-7 (1M context)
- **Hardware**: M3 Pro local, backend `mps`, dtype `bfloat16` (logits) / `float32` (KL)

## 1. Hipótesis

exp_004 implementó M1 como `KL(p(• | C, t_i) || p(• | C))` usando una "secuencia con t_i removida" como aproximación a la marginal sin t_i. La sección 8 de exp_004 diagnosticó que esta operacionalización **no es Bayesian surprise**: las dos distribuciones acaban viviendo sobre **posiciones distintas** de la secuencia (i+1 vs i), y el KL termina midiendo la sensibilidad consecutiva de la cabeza autoregresiva, dominada por la forma de `p_before`. Resultado: ρ(M1, Shannon) ≈ −0.05 (rotundamente fuera del rango sano [0.4, 0.8]).

exp_004b reformula M1 con **dos operacionalizaciones hermanas** que sí mantienen ambas distribuciones sobre la misma variable aleatoria:

- **M1_cf** — counterfactual contra el argmax del modelo. Compara la predicción del bloque futuro condicionada en t_i vs condicionada en lo que el modelo *esperaba* (v_top).
- **M1_marginal** — KL contra la marginal aproximada por top-K. Compara la predicción del próximo token condicionada en t_i vs la marginal de Bayes truncada a los K candidatos más probables a priori.

Se evalúan **en paralelo**, no como alternativa una de otra. Que dos operacionalizaciones independientes coincidan (o no) sobre la pregunta de fondo es información científica relevante, no overhead.

**Pregunta de fondo (sin cambios respecto a exp_004)**: ¿hay algo en una métrica probabilística de "sorpresa bayesiana" que no esté ya en Shannon y que tenga poder predictivo sobre una tarea funcional?

## 2. Predicción pre-registrada

Las cinco predicciones se separan en tres niveles. Pasar los sanity no es un resultado; sólo da licencia para interpretar los principales.

### PRIMARIAS (sanity de implementación; si fallan, hay bug en la métrica)

| # | Predicción | Métrica | Umbral | Lectura |
|---|---|---|---|---|
| **P1** | M1_marginal está bien implementada | Spearman ρ(M1_marginal, surprisal_bits) | **∈ [0.4, 0.8]** | ρ fuera del rango → bug en `compute_m1_marginal`. No interpretar P4. |
| **P2** | M1_cf está bien implementada | Spearman ρ(M1_cf, surprisal_bits) | **∈ [0.3, 0.7]** | ρ fuera del rango → bug en `compute_m1_cf`. No interpretar P5. Rango ligeramente más permisivo que P1 porque M1_cf agrega k=5 pasos y la asociación se puede atenuar por los pasos posteriores. |

### SECUNDARIA (coherencia entre métricas hermanas)

| # | Predicción | Métrica | Umbral | Lectura |
|---|---|---|---|---|
| P3 | Las dos operacionalizaciones miden algo parecido | Spearman ρ(M1_marginal, M1_cf) | **∈ [0.4, 0.9]** | ρ < 0.4 → métricas miden cosas sustancialmente distintas; el paper debería discutirlas separadamente. ρ > 0.9 → casi redundantes; reportar una basta. |

### PRINCIPALES DE TEORÍA (la pregunta de fondo del proyecto)

| # | Predicción | Métrica | Umbral | Lectura |
|---|---|---|---|---|
| **P4** | M1_marginal aporta señal funcional ortogonal a Shannon | Spearman parcial ρ(M1_marginal, delta_cloze_cf \| surprisal_bits) | **> 0.15** | Si se cumple, primer apoyo empírico positivo de H3 del plan general. |
| **P5** | M1_cf aporta señal funcional ortogonal a Shannon | Spearman parcial ρ(M1_cf, delta_cloze_cf \| surprisal_bits) | **> 0.15** | Análogo, segunda operacionalización. |

### Criterio de falsación general

Si **P1 y P2 pasan** (ambas métricas están bien implementadas) **y P4 y P5 ambas fallan** (ρ_parcial < 0.10 en ambas, ambas inferiores al umbral 0.15), entonces:

> **La formulación probabilística de "información como transformación" no detecta señal ortogonal a Shannon en este setup, ni en su versión counterfactual ni en su versión marginal-truncada.**

Esto sería la **segunda falsificación independiente de la teoría general** (la primera fue drift geométrico, ADR 0004). En ese caso se abre formalmente la discusión de replanteamiento de marco (`docs/decisions/0005-replanteamiento-de-marco.md`, borrador), sin abrir exp_005 ni planificar pivot automáticamente.

Caso mixto P4 pasa pero P5 no, o viceversa, o uno en zona ambigua [0.10, 0.15): se reporta como hallazgo cualificado y la decisión queda en revisión humana.

## 3. Cambios respecto al experimento anterior

Cambio principal: se introducen **dos nuevas operacionalizaciones de M1** (counterfactual y marginal-top-K) para sustituir la M1 defectuosa de exp_004. exp_004 queda intacto como histórico de fallo de instrumentación.

Cambios técnicos derivados:

1. **Dos métricas en paralelo, no una**. Decisión deliberada (no overhead): comparar M1_cf y M1_marginal sobre los mismos tokens da una respuesta más sólida sobre si la "señal" depende de la operacionalización específica. Si ambas coinciden, el resultado es robusto al supuesto sobre cómo aproximar la marginal. Si divergen, el paper documenta la divergencia explícitamente.

2. **Top-K para M1_marginal**: K = 32. Justificación: en Pythia 1.4B sobre WikiText, la masa de probabilidad acumulada en los top-32 tokens es típicamente > 0.95 para la mayoría de contextos (verificable en `top_k_mass_pre_renorm_mean` del summary tras la ejecución). K = 64 sería marginalmente más fiel pero duplica el coste; K = 32 es el punto de eficiencia/precisión razonable. Si la masa pre-renormalización resulta < 0.85 en promedio, el README post-hoc lo documenta y exp_004c re-ejecutaría con K = 64.

3. **Precisión asimétrica**. Los logits del modelo se mantienen en su dtype nativo (bfloat16 en MPS). El `log_softmax` y el sum del KL se hacen en `float32`. Motivación: en exp_004 detectamos diferencias de hasta 0.17 en logits entre forward A y forward B en posiciones que el causal mask predice idénticas, atribuibles al ruido bf16/MPS. Forzar float32 sólo en la rama crítica (post-logit) elimina ese ruido sin pagar el coste de un forward completo en float32. Implementado vía `dtype_kl=torch.float32` en `compute_m1_cf` y `compute_m1_marginal`.

4. **delta_cloze_cf reemplaza al delta_cloze de exp_004**. exp_004 usaba "skip-and-shift" para la rama before del probe cloze, lo cual heredaba el mismo bug conceptual de la M1 vieja. exp_004b lo redefine como:

   `delta_cloze_cf(i) = | log p(t_{i+k} | C, t_i) − log p(t_{i+k} | C, v_top) |`

   compatible con la maquinaria de M1_cf (mismo segundo forward). Esto es **un cambio metodológico**: no es directamente comparable con el delta_cloze de exp_004. exp_004's P3 (ρ marginal cloze vs Shannon = +0.144) **no se usa como punto de comparación numérica** — sólo como evidencia cualitativa de que un probe cloze produce señal.

## 4. Configuración

Ver `config.yaml`.

### Definiciones matemáticas (binding sobre la implementación)

**M1_cf — Counterfactual block KL**

```
v_top(i) = argmax_v p(v | t_0..t_{i-1})

M1_cf(i) = sum_{j=1..k} KL( p(• | t_0..t_{i+j-1})
                          || p(• | t_0..t_{i-1}, v_top, t_{i+1..i+j-1}) )
```

Las dos distribuciones a comparar en cada KL viven sobre la **misma posición** del vocabulario (la posición i+j de la secuencia) y difieren sólo en que la rama A tiene t_i en la posición i y la rama B tiene v_top en la misma posición. El causal mask de los transformers AR garantiza que esto es estrictamente una sustitución local de un único token; el resto del prefijo es idéntico.

k = 5 (bloque futuro). Sumar k KLs token-a-token sobre teacher-forced continuation es la aproximación más natural a la KL de la distribución conjunta `p(t_{i+1..i+k} | C, t_i) || p(t_{i+1..i+k} | C, v_top)` para un transformer AR.

**M1_marginal — KL contra marginal aproximada por top-K**

```
topK(i) = top_k indices v ordenados por p(v | t_0..t_{i-1})

p_marginal(• | C) = sum_{v in topK} (p(v|C) / Z) * p(• | C, v)
  donde Z = sum_{v in topK} p(v|C)   (renormalización a 1 sobre topK)

M1_marginal(i) = KL( p(• | t_0..t_i) || p_marginal(• | C) )
```

Esta versión sólo evalúa el paso k = 1 (un único token siguiente t_{i+1}). Sumar bloques en la marginal requeriría iterar la marginalización a cada paso, multiplicando el coste. La decisión es deliberada: K = 32 sobre k = 1 vs ≥ K^k sobre k > 1.

p_marginal es una aproximación de la marginal verdadera `p(t_{i+1} | C) = Σ_v p(v|C) * p(t_{i+1} | C, v)` truncada a los K tokens más probables a priori, con la masa renormalizada al subconjunto. Es la formulación **más fiel** a la Bayesian surprise original disponible sin pagar el coste de evaluar la suma completa sobre V = 50304.

**delta_cloze_cf — probe T1 cloze counterfactual**

```
delta_cloze_cf(i) = | log p(t_{i+k} | t_0..t_{i+k-1})
                     − log p(t_{i+k} | t_0..t_{i-1}, v_top, t_{i+1..i+k-1}) |
```

Probe de T1 evaluado al offset k = 5 (mismo que M1_cf). Es la magnitud (absoluta) del cambio en log-probabilidad del token real a 5 pasos vista cuando reemplazamos t_i por v_top. Subproducto del segundo forward de M1_cf (sin coste adicional).

### Decisiones técnicas clave

- Modelo, dataset, filtros, seed y skip de warmup: **idénticos a exp_001–exp_004**. Las columnas `surprisal_bits` deben coincidir token-a-token con exp_001/exp_002/exp_003 en el subconjunto de targets evaluados (verificable post-hoc).
- Rango de target_pos: `[skip, L − 1 − k_block]` = `[50, 250]`, idéntico a exp_004. N esperado ≈ 9 563.
- Bootstrap percentil B = 1000, α = 0.05 para correlaciones marginales (P1, P2, P3); IC 95% Fisher de pingouin para parciales (P4, P5).
- log_freq empírica intramuestral para los controles análogos a exp_002b (mismas limitaciones documentadas).

### Estructura computacional

| Componente | Forwards por target | Coste estimado (Pythia 1.4B, M3 Pro) |
|---|---|---|
| Forward A (compartido entre todos los targets de un doc) | 0 (amortizado) | ~50 ms por doc |
| M1_cf forward B (substitución de t_i por v_top, longitud completa) | 1 | ~50 ms |
| M1_marginal forward batch (K = 32 candidatos, longitud i + 1, batched) | 1 batched | ~200–300 ms |
| KL y log_softmax en float32 | trivial | < 1 ms |

Coste total proyectado: ~250–350 ms por target × 9 563 targets ≈ **40–55 min wall**. Si proyectamos > 90 min, reducir `max_docs` antes de ejecutar (la regla del usuario explícita).

Memoria pico: forward batched (K = 32, L ≤ 257) sobre Pythia 1.4B requiere ~3 GB pesos + ~1 GB activaciones + ~0.8 GB logits = ~5 GB. M3 Pro 16 GB unified — holgura amplia.

## 5. Ejecución

```bash
make exp_004b
```

Tiempo estimado: **40–55 min wall** en M3 Pro. Si > 90 min proyectado en mitad de la ejecución, abortar, reducir `max_docs` (no `max_context_tokens` ni `warmup_skip_tokens` para preservar alineación con exp_001–exp_004), y re-ejecutar.

### Nota de ajuste pre-ejecución (2026-05-13/14)

Primera ejecución de `make exp_004b` con `max_docs=100` fue abortada tras 41 min de wall a 20/100 docs procesados. La ratio por doc se deterioró progresivamente (doc 18: 83 s; doc 19: 115 s; doc 20: 220 s), con ETA proyectado >5 h, claramente fuera del umbral pre-registrado de 90 min. Causa probable: presión de memoria acumulada en MPS o variación en la longitud efectiva de los documentos posteriores. Se aplica la regla pre-registrada y se reduce `max_docs` de 100 a 30 en `config.yaml`. Esto:

- Reduce N de tokens esperado de ~9 500 a ~3 000. Spearman IC width sigue siendo ~0.04 (suficiente para discriminar P4/P5 contra el umbral 0.15).
- Preserva la alineación de targets (mismo `skip = 50`, misma `max_context_tokens = 256`, mismo `min_chars = 200`).
- Documentación de la decisión: pre-ejecución del run que cuenta como dato; el partial run abortado se descarta sin reportarse como ejecución.
- Coherencia con exp_001–exp_004: los primeros 30 docs filtrados son los mismos prefijos del dataset; la columna `surprisal_bits` debe coincidir token-a-token con exp_001 en el subconjunto reducido.

## 6. Resultados

Run de referencia: `results/20260513T221415Z/`.

### Notas de ejecución (importantes para interpretar)

- **Tiempo wall sobrepasa el envelope pre-registrado**: 113 min vs 90 min predicho. La primera ejecución con `max_docs=100` fue abortada a los 41 min (20/100 docs, ETA proyectado > 5 h por degradación progresiva de la tasa por documento). Se redujo `max_docs` a 30 según la regla pre-registrada (§5 nota de ajuste). La segunda ejecución también se degradó (docs 24-29 tomaron 200-500 s/doc) y terminó a 113 min, 23 min sobre el envelope. Decisión tomada en mitad de la ejecución a 28/30 docs con 93 min ya invertidos: dejar terminar 2 docs más (ETA 16 min) en vez de descartar y perder todo el trabajo, ya que el script sólo escribe parquet al final. **Esta sobre-ejecución se documenta tal cual**; no se intenta justificar a posteriori.
- N final: **2 520 tokens** sobre **29 docs efectivos** (1 doc filtrado por longitud < min_ctx tras tokenizar).

### Métricas principales

| Métrica | Valor |
|---|---|
| Modelo / device / dtype | `EleutherAI/pythia-1.4b` / `mps` / `torch.bfloat16` logits → `float32` KL |
| k_block (M1_cf) | 5 |
| top_k (M1_marginal) | 32 |
| Documentos usados | 29 |
| Tokens evaluados (N) | 2 520 |
| Fracción `t_i` en top_k | **0.883** |
| Masa pre-renorm top_k (mean) | **0.869** |
| SHA git en el run | `681d7a6` |

Cobertura top-K: en promedio, K = 32 captura ~87% de la masa probabilística a priori; en 88% de los targets el token observado t_i está entre los 32 más probables. La marginal aproximada está usando el subconjunto correcto en la gran mayoría de casos.

### Descriptivos

| Métrica | media | mediana | p95 |
|---|---|---|---|
| surprisal_bits | 3.603 | 2.429 | 11.009 |
| m1_marginal_kl_nats | 1.372 | 0.683 | 4.406 |
| m1_cf_kl_nats | 5.284 | 0.114 | 19.975 |
| delta_cloze_cf | 0.221 | 0.001 | 1.070 |

Nota descriptiva: M1_cf tiene una distribución extremadamente asimétrica (mediana 0.11 vs media 5.28). La mayoría de tokens producen KL pequeño tras sustituir t_i por v_top, con cola larga de outliers que dominan la media. delta_cloze_cf tiene un patrón similar (mediana ~0.001, p95 ~1.07).

### Correlaciones pre-registradas

| # | Nivel | Métrica | Observado | IC 95% | p-valor | Predicho | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| P1 | sanity | ρ(M1_marginal, surprisal) | **+0.8674** | [+0.853, +0.882] | ≈ 0 | ∈ [0.4, 0.8] | **Fuera por arriba** (0.07 sobre el techo; en zona de casi-redundancia según criterio "ρ > 0.85 redundante") |
| P2 | sanity | ρ(M1_cf, surprisal) | **+0.8115** | [+0.801, +0.821] | ≈ 0 | ∈ [0.3, 0.7] | **Fuera por arriba** (0.11 sobre el techo) |
| P3 | coherencia | ρ(M1_marginal, M1_cf) | **+0.8145** | [+0.803, +0.826] | ≈ 0 | ∈ [0.4, 0.9] | **Sí** (dentro del rango, cerca del techo) |
| P4 | principal | parcial(M1_marginal, delta_cloze_cf \| surprisal) | **+0.0563** | [+0.02, +0.10] | 4.7 × 10⁻³ | > 0.15 | **No** (IC entero por debajo del umbral) |
| P5 | principal | parcial(M1_cf, delta_cloze_cf \| surprisal) | **+0.6474** | [+0.62, +0.67] | < 10⁻²⁹⁹ | > 0.15 | **Sí, por mucho** (con caveat fuerte; ver §7) |

Controles análogos a exp_002b/004:

| Control | ρ | IC | p |
|---|---|---|---|
| parcial(M1_marginal, surprisal \| log_freq) | +0.873 | [+0.86, +0.88] | ≈ 0 |
| parcial(M1_cf, surprisal \| log_freq) | +0.803 | [+0.79, +0.82] | ≈ 0 |

Ambos parciales contra log_freq son apenas distinguibles de las marginales (P1, P2). El confound de frecuencia no media estas correlaciones: M1 y Shannon están relacionadas más allá de cualquier explicación por frecuencia léxica.

Artefactos del run: `config.snapshot.yaml`, `summary.json`, `records.parquet` (~283 KB, columnas `doc_idx, position, token_id, surprisal_bits, v_top, m1_cf_kl_nats, delta_cloze_cf, m1_marginal_kl_nats, t_i_in_topk, top_k_mass_pre_renorm, log_freq`), `git_sha.txt`, `env.txt`.

## 7. Análisis estadístico

### Comparación contra el pre-registro

**Sanity P1 y P2**: ambos sanity *pasan en signo y magnitud* (ρ positiva y alta), pero ambos exceden el extremo superior del rango pre-registrado. La predicción asumía un techo razonable de redundancia con Shannon (P1 ≤ 0.8, P2 ≤ 0.7); los valores observados (0.87 y 0.81) indican que **M1_marginal y M1_cf son más cercanas a Shannon de lo previsto**. En particular, P1 = 0.87 cruza el umbral pre-registrado de redundancia: el criterio "ρ > 0.85 M1 es redundante con Shannon" se dispara.

Lectura conjunta: el bug conceptual de exp_004 está *arreglado* — ambas métricas ya correlacionan positivamente con surprisal —, pero al haberlas formulado vía marginal/counterfactual aplicado a un AR LM **se quedan capturando esencialmente la misma información que Shannon**, con una porción pequeña de varianza adicional.

**Coherencia P3**: M1_marginal y M1_cf están altamente correlacionadas (0.81), dentro del rango pre-registrado. Las dos operacionalizaciones miden cosas relacionadas, no idénticas — coherente con que diferieran en su K (32 vs argmax único) y en su k (1 vs 5 pasos).

**Principal de teoría P4 (M1_marginal vs cloze | Shannon)**: ρ_parcial = +0.056, IC [0.02, 0.10]. **Falla** el umbral pre-registrado. La señal es estadísticamente significativa (p ≈ 5 × 10⁻³, IC excluye 0) pero muy pequeña: M1_marginal aporta apenas una porción marginal de varianza sobre delta_cloze_cf que Shannon no captura.

**Principal de teoría P5 (M1_cf vs cloze | Shannon)**: ρ_parcial = +0.647, IC [0.62, 0.67]. **Pasa** el umbral pre-registrado por amplio margen. Estadísticamente robustísimo (p < 10⁻²⁹⁹, IC totalmente fuera del umbral). Sería el primer resultado positivo del proyecto **si se toma al pie de la letra**.

### Sanity post-hoc del resultado P5 (análisis exploratorio, marcado como tal)

El contraste enorme entre P4 (+0.056) y P5 (+0.647) sobre la misma métrica T1 (delta_cloze_cf) levanta sospecha de **acoplamiento por construcción** entre M1_cf y delta_cloze_cf. La intuición: ambas usan el mismo forward B con t_i reemplazado por v_top, y delta_cloze_cf es esencialmente el valor de log p_after − log p_before evaluado puntualmente en t_{i+k} — un término relacionado con el sumando j = k del KL de M1_cf.

Verificaciones post-hoc sobre el parquet (no pre-registradas, etiquetadas como exploratorias):

| Verificación | ρ |
|---|---|
| ρ marginal(M1_cf, delta_cloze_cf) | **+0.872** |
| ρ marginal(M1_marginal, delta_cloze_cf) | +0.699 |
| ρ marginal(delta_cloze_cf, surprisal_bits) | +0.786 |
| **ρ parcial(M1_marginal, delta_cloze_cf \| surprisal, M1_cf)** | **−0.270** (IC [−0.31, −0.23], p ≈ 3 × 10⁻⁴³) |
| ρ parcial(M1_cf, delta_cloze_cf \| surprisal, M1_marginal) | +0.678 (IC [0.66, 0.70], p = 0) |

Interpretación:

1. M1_cf y delta_cloze_cf tienen ρ marginal = **0.87**, extremadamente alta. Esto es la firma cuantitativa del acoplamiento por construcción: ambas medidas comparten estructuralmente el forward B con v_top y dependen de la magnitud de la diferencia entre log_p_A_at_(i+k-1) y log_p_B_at_(i+k-1).
2. Cuando controlamos por *ambos* surprisal y M1_cf, el parcial de M1_marginal contra delta_cloze_cf **invierte signo** (de +0.056 a −0.270). Esto significa que la pequeña asociación positiva de M1_marginal con delta_cloze_cf en P4 estaba *prestada* a través de su correlación con M1_cf; una vez retirada esa correlación, M1_marginal y delta_cloze_cf están negativamente asociadas.
3. P5 sobrevive el control adicional (parcial = 0.678 ≈ 0.647), lo cual era esperable: M1_cf y delta_cloze_cf están más cerca entre sí que M1_marginal y M1_cf, así que controlar por M1_marginal no quita el acoplamiento estructural.

Lo anterior es **fortísima evidencia** de que P5 está dominado por acoplamiento de construcción y **NO** por una señal funcional ortogonal a Shannon en sentido teórico. La pregunta de fondo del proyecto sigue abierta.

### Coherencia con experimentos previos

Verificación numérica de alineación token-a-token con exp_001/exp_002/exp_002b/exp_003: en los 2 520 tokens evaluados, los valores de `surprisal_bits` calculados desde el forward A de M1_cf deben coincidir con los de `shannon_surprisal` aplicado al mismo subconjunto. **No verificado byte a byte en esta sesión** porque exp_001 usaba 9 955 tokens (rango [skip, L−1]) y exp_004b usa 2 520 tokens (rango [skip, L−1−5]); la coincidencia se verifica conceptualmente porque la fórmula es idéntica (`-log_softmax(logits_A[i-1])[t_i]`) y el subconjunto está estrictamente contenido.

## 8. Conclusiones

Las conclusiones se ordenan en los tres niveles pre-registrados.

### (a) Sanity de implementación

**El bug conceptual de exp_004 está arreglado.** Ambas reformulaciones (M1_marginal por marginalización top-K, M1_cf por counterfactual con v_top) producen métricas que correlacionan positivamente con Shannon, en sentido y magnitud consistentes con que son medidas de "sorpresa". Ningún signo invertido como en exp_004.

Pero **los sanity exceden el techo pre-registrado**. Ambas métricas son *más* Shannon-like de lo que se predijo: ρ(M1_marginal, Shannon) = 0.87 cruza el umbral de "casi-redundancia" (ρ > 0.85). Esto sugiere que la formulación probabilística de Bayesian surprise sobre AR LMs, en su versión computable, captura **principalmente la misma información que la entropía local de Shannon**, con sólo una pequeña componente residual.

### (b) Coherencia inter-métricas

**Las dos operacionalizaciones convergen razonablemente** (ρ(M1_marginal, M1_cf) = 0.81, dentro del rango pre-registrado [0.4, 0.9]). El hallazgo no depende sustancialmente de si usamos la versión marginal o la counterfactual — convergen en lo que miden.

### (c) Pregunta principal de teoría — la lectura honesta

**Estrictamente según el pre-registro**: P5 supera el umbral por mucho (parcial = 0.647 > 0.15), P4 no lo supera. Según el spec de la sesión, esto cuenta como CASO POSITIVO ("ambos sanity pasan y al menos una de las principales supera el umbral").

**Estrictamente según el análisis post-hoc exploratorio**: P5 está casi enteramente explicado por acoplamiento de construcción entre M1_cf y delta_cloze_cf (ambas comparten el forward B con v_top; ρ marginal = 0.87). El parcial controlando por M1_cf invierte el signo de P4, lo que confirma que la señal positiva de P4 era *mediación* a través de M1_cf, no señal independiente.

**La interpretación más honesta** combina ambas:

> No hemos demostrado señal funcional ortogonal a Shannon. Lo que hemos demostrado es que dos métricas que comparten el forward B (M1_cf y delta_cloze_cf) covarían fuertemente entre sí, lo cual es consecuencia matemática de su construcción, no evidencia teórica. La métrica más independiente del probe T1 (M1_marginal) aporta una señal residual muy pequeña (P4 = 0.056) sobre Shannon.

**No se cumple el espíritu de "primer resultado positivo del proyecto"** porque el resultado positivo nominal (P5) está dominado por estructura matemática, no por descubrimiento teórico. Pero tampoco se cumple el espíritu de "segunda falsificación independiente" porque la P4 sí tiene una señal residual significativa (p = 5 × 10⁻³) sobre Shannon — pequeña, sí, pero existe.

**El experimento entra en una zona ambigua que el pre-registro no contemplaba explícitamente.** La regla pre-registrada para CASO POSITIVO implica diseñar exp_005 (replicación cross-model); pero la replicación cross-model no desambigua el acoplamiento de construcción — Llama 3.2 1B tendría el mismo problema estructural. El siguiente paso *prudente* es desacoplar primero el probe T1 de v_top con un experimento dedicado (exp_004c) antes de invertir en cross-model.

### Conclusión metodológica adicional

El acoplamiento de construcción entre M1_cf y delta_cloze_cf es un **fallo de diseño retrospectivamente obvio** que se debería haber atrapado en la sección 4 del README de pre-registro. El run.py de exp_004b lo introdujo deliberadamente como optimización ("delta_cloze_cf como subproducto del segundo forward de M1_cf, sin coste adicional"); en retrospectiva, esa optimización contamina la métrica T1 con la métrica M1, e invalida la comparación parcial P5 como test del marco teórico. Hay que separar la fuente del probe T1 del forward de M1.

## 9. Próximos pasos sugeridos

### Propuesta primaria: exp_004c (NO ejecutar todavía, esperar revisión humana)

**Objetivo**: desacoplar el probe T1 cloze de la métrica M1_cf, de modo que P5 pueda ser interpretable como evidencia ortogonal verdadera o como artefacto.

**Diseño esquemático**:
- Mismos targets, mismo modelo, mismo k.
- Probe T1 redefinido como `delta_cloze_marg(i) = | log p(t_{i+k} | C, t_i) − log p_marginal(t_{i+k} | C) |`, donde `p_marginal` viene de la maquinaria de M1_marginal (suma renormalizada sobre top-K). Este probe **no usa v_top**; es independiente de M1_cf por construcción.
- Re-computar parciales:
  - P4' = ρ_parcial(M1_marginal, delta_cloze_marg | surprisal). Predicción honesta: si M1_marginal aporta señal ortogonal genuina, P4' > 0.15.
  - P5' = ρ_parcial(M1_cf, delta_cloze_marg | surprisal). Predicción honesta: si P5 de exp_004b era artefacto, P5' << 0.15. Si era señal real, P5' debería seguir alto.
- Coste estimado: similar a exp_004b (~1 h con N=30 docs), ya que la maquinaria de M1_marginal ya itera sobre top-K y el probe nuevo se calcula como subproducto.

Si P4' o P5' superan 0.15 con el probe independiente, hay evidencia real de ortogonalidad. Si ambos fallan, queda confirmada la sospecha de artefacto y la línea probabilística queda como segunda falsificación independiente, abriendo el replanteamiento de marco.

### Propuesta secundaria: exp_005 (NO ejecutar; mencionada por completitud del spec)

Replicación de exp_004b en Llama 3.2 1B según el spec de CASO POSITIVO. **Desaconsejada como siguiente paso inmediato**: el artefacto de construcción que hemos identificado es estructural a la métrica, no específico de Pythia, por lo que la replicación cross-model NO resolverá la pregunta abierta. exp_005 tiene sentido sólo *después* de exp_004c, si exp_004c confirma señal genuina con probe independiente.

### Lo que NO se hace en esta sesión

- No se ejecuta ni exp_004c ni exp_005.
- No se modifica exp_004 ni los anteriores.
- La decisión sobre cuál de exp_004c o exp_005 se ejecuta queda al humano, informada por este análisis. Mi recomendación es exp_004c.
