# Proyecto de investigación de Abit (Identy Labs)

# exp_004b — Bayesian surprise reformulada (counterfactual + marginal top-K)

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: `[pendiente]`
- **Autor**: Ezequiel
- **Claude Code model**: `[rellenar tras ejecutar]`
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

## 6. Resultados

`[rellenar tras ejecutar]`

## 7. Análisis estadístico

`[rellenar tras ejecutar; reportar P1–P5 con IC, categorizadas por nivel (sanity vs principal); verificar coherencia token-a-token de surprisal con exp_001]`

## 8. Conclusiones

`[rellenar tras ejecutar; tres niveles de conclusión: (a) sanity, (b) coherencia inter-métricas, (c) pregunta principal de teoría]`

## 9. Próximos pasos sugeridos

`[rellenar tras ejecutar; dependiente del resultado; NO planificar exp_005 automáticamente]`
