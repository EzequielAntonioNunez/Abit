# Proyecto de investigación de Abit (Identy Labs)

# Diseño Experimental

**Documento técnico complementario al Plan de Investigación**
**Versión**: 0.1
**Estado**: pre-registro experimental

---

## 1. Sujetos experimentales (modelos)

Todos open-weights, ejecutables localmente o en GPU comercial.

| Modelo | Parámetros | Razón de inclusión |
|--------|-----------|---------------------|
| Gemma 2 2B | 2.0B | Base local en M3 Pro, alto rendimiento por parámetro |
| Llama 3.2 1B | 1.2B | Comparación de escala pequeña |
| Llama 3.2 3B | 3.2B | Comparación de escala media |
| Pythia 1.4B | 1.4B | Tiene checkpoints intermedios, útil para análisis evolutivo |
| Pythia 2.8B | 2.8B | Control adicional |

Cargados vía HuggingFace `transformers` con `torch.bfloat16` y backend `mps` en M3, o CUDA en cloud.

**Justificación de exclusión**: modelos > 7B se descartan en fase inicial por coste. Si los resultados son sólidos, escalado posterior a Mistral 7B / Llama 8B vía Modal o Runpod.

---

## 2. Datasets

Todos públicos, licencias compatibles, acceso reproducible vía HuggingFace `datasets`.

| Dataset | Uso | N tokens objetivo |
|---------|-----|---|
| WikiText-103 | Control general, distribución natural | 50.000 |
| LAMBADA | Predicción dependiente de contexto largo | 10.000 |
| BLiMP | Pares mínimos gramaticales (control de manipulación) | Todos los pares |
| Counterfact (Meng et al.) | Hechos factuales editables, ground truth de "cambio de creencia" | Todos |
| arXiv abstracts subset | Alta densidad conceptual | 20.000 |

**Selección de tokens objetivo dentro de cada texto**: aleatoria estratificada por log-frecuencia (deciles), no curada. Seed fijo `42`.

---

## 3. Métricas a comparar

### M0 — Shannon surprisal (baseline)
```
M0(t_i | C) = -log_2 p(t_i | C)
```
Obtenido del logit del modelo. Implementación trivial, sirve como control.

### M1 — Bayesian surprise sobre bloque futuro
```
M1(t_i | C, k) = D_KL( P(t_{i+1:i+k} | C, t_i) || P(t_{i+1:i+k} | C) )
```
Aproximación práctica: KL token-a-token sobre los siguientes `k` pasos (probar k ∈ {1, 5, 20}).
Coste: 2 forward passes por token objetivo. Manejable.

### M2 — Activation drift (norma L2)
```
M2_l(t_i) = || h_l(C, t_i) - h_l(C) ||_2 / sqrt(d_l)
```
Para cada capa `l`. Métricas derivadas:
- M2_last: solo última capa
- M2_mean: promedio sobre capas
- M2_max: máximo sobre capas

Normalización por dimensión `d_l` para comparabilidad inter-capa.

### M3 — Attention entropy delta
```
M3(t_i) = mean over heads, layers of | H(att_l,h(C, t_i)) - H(att_l,h(C)) |
```
Mide cambio en patrón de atención.

### M4 — Cosine drift en embedding final
```
M4(t_i) = 1 - cos( h_L(C, t_i), h_L(C) )
```
Variante normalizada de M2, escala-invariante.

### M5 — Effective dimension drift (preliminar)
Cambio en la dimensión efectiva (participation ratio) de las activaciones. Exploratoria.

---

## 4. Tareas downstream (ground truth de "impacto")

Sin estas, las métricas serían autocorrelaciones del modelo consigo mismo. Necesitamos ancla externa.

### T1 — Cloze prediction shift
Para cada token objetivo `t_i`, comparar:
- `P(y* | C)` vs. `P(y* | C, t_i)` para una respuesta `y*` evaluada en un contexto posterior.
- Métrica: `Δ_cloze = | log P(y*|C,t_i) - log P(y*|C) |`

### T2 — Factual QA shift (Counterfact)
Tokens que introducen edición factual deberían producir cambios grandes en respuestas factuales relacionadas.

### T3 — Behavioral consistency
¿Cambia la respuesta del modelo a la misma pregunta cuando `t_i` está presente vs. ausente? Medido por log-likelihood ratio.

---

## 5. Diseño estadístico

### Tamaños de muestra
- N tokens por dataset: 10.000 mínimo
- N pares emparejados para H1: 5.000
- N modelos: 5
- Total estimado de mediciones: ~250.000 tokens × 6 métricas × 5 modelos = 7.5M observaciones

### Análisis primario
1. **Correlación de Spearman** entre cada métrica Mi y cada tarea downstream Tj
2. **Regresión múltiple**: `Tj ~ M0 + M1 + M2_mean + M3 + M4 + controles`
3. **Coeficientes parciales**: aislar contribución de cada métrica controlando por surprisal Shannon

### Controles obligatorios
- Log-frecuencia del token objetivo
- Longitud del contexto
- Posición en el documento
- Identidad del modelo
- Identidad del dataset

### Inferencia estadística
- Bootstrap (B = 10.000) para intervalos de confianza
- Corrección Bonferroni o BH-FDR para comparaciones múltiples (6 métricas × 3 tareas × 5 modelos = 90 tests)
- Umbral pre-registrado: `p < 0.01` corregido

### Análisis de robustez
- Repetir con seeds {42, 1337, 2024, 7, 99}
- Permutation test: shuffle tokens objetivo, verificar que efectos desaparecen
- Subset analysis: ¿se mantiene en arXiv vs. WikiText?

---

## 6. Confounds previstos y mitigación

| Confound | Riesgo | Mitigación |
|----------|--------|------------|
| Frecuencia léxica | Tokens raros → Shannon alto Y activation drift alto | Estratificar por decil de frecuencia |
| Longitud de contexto | Contextos largos → métricas más estables | Fijar ventanas de 256 tokens |
| Posición en texto | Tokens iniciales más predecibles | Excluir primeros 50 tokens |
| Tokenización | BPE puede dividir palabras de forma artificial | Agrupar a nivel de palabra cuando sea posible |
| Numerical precision | bfloat16 puede afectar KL pequeñas | Verificar con float32 en subset |
| Selection bias en pares H1 | Emparejar por surprisal puede seleccionar tokens atípicos | Análisis sobre distribución completa, no solo emparejados |

---

## 7. Pre-registro de predicciones cuantitativas

**Antes de correr el experimento principal, predicción esperada**:

- Correlación M0 vs. M2_mean: r ∈ [0.3, 0.6] (no es trivial pero no es independiente)
- Correlación M0 vs. T1 (cloze shift): r ∈ [0.2, 0.5]
- Correlación M1 vs. T1: r ∈ [0.4, 0.7] (mejor que M0)
- Varianza residual explicada por M2 tras controlar M0+M1: ≥ 5%
- Estabilidad cross-model del ranking de métricas: Kendall τ > 0.6

Si los resultados están todos dentro de estos rangos, son consistentes con H2-H3.
Si M1 no supera a M0, H2 falsada.
Si M2 residual < 5%, H3 falsada.

---

## 8. Análisis exploratorio (post-hoc, marcado como tal)

Tras los análisis confirmatorios, explorar:
- Estructura factorial de las métricas (PCA)
- Clustering de tokens por perfil de métricas
- Análisis capa-por-capa del activation drift (¿hay capas "más informativas"?)
- Comparación inter-modelo: ¿los modelos coinciden en qué tokens son "significativamente sorpresivos"?

**Cualquier hallazgo aquí no se reporta como confirmatorio.**

---

## 9. Replicabilidad

- Código en GitHub, MIT
- Seeds fijos en todo
- Versiones de modelos pinneadas por SHA en HuggingFace
- Datasets cacheados con hash MD5 verificado
- Dependencias en `pyproject.toml` con versiones exactas
- Resultados intermedios versionados (DVC o git-lfs)
- Logs de cada run con configuración completa (Hydra)
- Tracking experimental en MLflow autoalojado o Weights & Biases

---

## 10. Cronograma experimental detallado

| Semana | Tarea | Output verificable |
|--------|-------|---------------------|
| 1 | Setup pipeline, cargar modelos, replicar surprisal Shannon en WikiText | Tabla con perplejidad replicada vs. valores publicados |
| 2 | Implementar M1, M2, M3, M4, tests unitarios | Suite de tests pasando |
| 3 | Experimento piloto N=1000 tokens, 1 modelo | Análisis preliminar de correlaciones |
| 4 | Experimento principal: WikiText + LAMBADA, 3 modelos | Tablas de resultados parciales |
| 5 | Experimento principal: BLiMP + Counterfact + arXiv | Tablas completas |
| 6 | Análisis estadístico, controles, bootstrap | Figuras finales, IC, p-valores corregidos |
| 7 | Replicación cross-model, análisis de robustez | Tabla de estabilidad |
| 8 | Escritura, decisión go/no-go preprint | Manuscrito v0.1 |
