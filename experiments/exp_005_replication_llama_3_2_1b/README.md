# Proyecto de investigación de Abit (Identy Labs)

# exp_005 — Replicación cross-architecture en Llama 3.2 1B

## Metadatos

- **Fecha de diseño**: 2026-05-14
- **Fecha de ejecución**: `[pendiente — pre-registrado SIN ejecutar; espera autorización humana]`
- **Autor**: Ezequiel
- **Claude Code model**: `[rellenar tras ejecutar]`
- **Hardware**: M3 Pro local, backend `mps`, dtype `bfloat16` (logits) / `float32` (KL)

## 1. Hipótesis

exp_004b produjo un **CASO POSITIVO según el pre-registro literal**: P5 (`ρ_parcial(M1_cf, delta_cloze_cf | surprisal) = +0.647`, IC [0.62, 0.67]) supera el umbral pre-registrado de 0.15 por amplio margen. Bajo la regla pre-registrada de la sesión, el siguiente paso es **replicar el resultado positivo en un segundo modelo** para descartar que sea artefacto de Pythia.

Hipótesis pre-registrada: si la señal de exp_004b refleja una propiedad real de "información ortogonal a Shannon" detectable por M1_cf en LMs autoregresivos, debe **replicarse en Llama 3.2 1B sobre el mismo dataset (WikiText-103 validation, mismos targets)**. Si no replica, la señal era artefacto de Pythia y la línea probabilística queda en duda en el sentido cross-architecture.

### Caveat de honestidad — leer antes de ejecutar

El análisis exploratorio post-hoc de exp_004b (§7) detectó que la P5 está **probablemente dominada por acoplamiento de construcción** entre M1_cf y delta_cloze_cf (ambas comparten el forward B con t_i reemplazado por v_top; ρ marginal entre ellas = 0.87). exp_005 es replicación literal de exp_004b en otro modelo, por lo que **hereda el acoplamiento estructural**: si la señal de P5 era artefacto en Pythia, también será artefacto en Llama, con la misma forma.

En otras palabras: exp_005 puede confirmar que el resultado es **reproducible** (lo cual es útil), pero **no puede desambiguar** si es señal teórica o artefacto matemático. La desambiguación requiere **exp_004c**, que redefine delta_cloze como `delta_cloze_marg` (independiente de v_top), y que está propuesto en `experiments/exp_004b_*/README.md` §9.

**Recomendación explícita al humano que autorice la ejecución**: ejecutar exp_004c **antes** de exp_005. Si exp_004c muestra que P5' con probe independiente sigue siendo alto, exp_005 vuelve a tener sentido como replicación cross-architecture genuina. Si exp_004c muestra que P5' colapsa, la señal de exp_004b era artefacto y exp_005 sería confirmar un artefacto en otro modelo — coste sin valor científico.

Esta sección queda escrita ANTES de cualquier ejecución para que el contraste con el pre-registro estricto del usuario sea visible.

## 2. Predicción pre-registrada

### Hipótesis de replicación (la pregunta del experimento)

| # | Predicción | Métrica | Umbral | Lectura |
|---|---|---|---|---|
| **R1** | M1_cf replica el resultado positivo | Spearman parcial ρ(M1_cf, delta_cloze_cf \| surprisal) en Llama 3.2 1B | **> 0.15** | Si pasa, el resultado de exp_004b es reproducible en otra arquitectura. Si falla, la señal era específica de Pythia. |
| R2 | M1_marginal mantiene su comportamiento (señal residual pequeña) | Spearman parcial ρ(M1_marginal, delta_cloze_cf \| surprisal) en Llama 3.2 1B | en [−0.10, +0.20] | Consistente con exp_004b P4 = +0.056. |

### Hipótesis de coherencia inter-modelo (sanity de la replicación)

| # | Predicción | Métrica | Umbral | Lectura |
|---|---|---|---|---|
| R3 | Sanity sigue funcionando | Spearman ρ(M1_marginal, surprisal) en Llama | ∈ [0.5, 0.95] | Llama 3.2 1B tiene arquitectura distinta a Pythia, pero ambas son AR LMs; rango más amplio para acomodar variación de arquitectura. |
| R4 | Sanity M1_cf análogo | Spearman ρ(M1_cf, surprisal) en Llama | ∈ [0.4, 0.9] | Análogo a R3. |

### Criterios de falsación

- Si **R1 falla** (parcial M1_cf vs cloze | Shannon < 0.10 en Llama): el resultado positivo de exp_004b NO replica cross-architecture. Dos interpretaciones posibles:
  1. La señal de exp_004b era artefacto específico de Pythia.
  2. La señal era acoplamiento de construcción, que en Llama se manifiesta distinto por razones del tokenizer o de la geometría del unembedding.
  En cualquier caso, no queda evidencia robusta de "información ortogonal a Shannon".
- Si **R3 o R4 fallan** (sanity ≪ 0.4): hay problema de implementación en Llama (tokenizer, dtype, gating) y no se puede interpretar R1/R2.

### Criterio de éxito

- R1 pasa Y los sanity (R3, R4) pasan → primer resultado positivo replicado cross-architecture. exp_006 entraría a discutir desambiguación del artefacto y exp_004c (independencia del probe T1) pasa a ser **necesario** para certificar interpretabilidad.

## 3. Cambios respecto al experimento anterior

Único cambio respecto a exp_004b: el modelo. `EleutherAI/pythia-1.4b` → `meta-llama/Llama-3.2-1B`. Todo lo demás (dataset, filtros, `max_docs=30`, `k=5`, `top_k=32`, métricas, parciales) es idéntico para que la comparación cross-architecture sea limpia.

## 4. Configuración

Ver `config.yaml`.

### Decisión: Llama 3.2 1B vs alternativa Pythia 2.8B

`meta-llama/Llama-3.2-1B` está marcado como gated en HuggingFace y requiere autenticación con `HF_TOKEN`. En la sesión de exp_001 ya se encontró que Gemma 2 2B estaba gated y se sustituyó por Pythia 1.4B (ADR 0002).

**Plan A (preferido si el token está disponible)**: ejecutar con `meta-llama/Llama-3.2-1B`. Es la arquitectura más distinta a Pythia 1.4B disponible en escala ≈ 1B (RMSNorm, GQA, SwiGLU, RoPE, tokenizer distinto).

**Plan B (fallback)**: si Llama está gated y no hay credenciales, sustituir por `EleutherAI/pythia-2.8b` (open weights, escala ligeramente mayor, **misma familia arquitectónica que Pythia 1.4B** — por lo tanto no es replicación cross-architecture genuina, sino cross-scale dentro de la misma familia). Documentar como sustitución análoga a ADR 0002 en una nueva ADR 0005 antes de ejecutar, NO en este README.

### Estructura computacional

Mismo coste por target que exp_004b (~300 ms con M1_marginal batched K=32 más M1_cf forward B). Llama 3.2 1B tiene 16 capas (vs Pythia 1.4B con 24), por lo que cada forward debería ser ~30% más rápido. N esperado ≈ 2 500 tokens con `max_docs=30` (mismo subconjunto que exp_004b).

Estimación: 50-80 min wall en M3 Pro. Si tras 90 min se proyecta > 120 min, abortar y reducir N, idéntica regla que exp_004b.

### Decisiones técnicas mantenidas idénticas a exp_004b

- Precisión asimétrica: logits en dtype del modelo (bf16/MPS), log_softmax y KL en float32.
- M1_cf con k = 5, M1_marginal con top_k = 32.
- delta_cloze_cf como probe T1 (con el caveat de acoplamiento de construcción documentado en §1).
- Bootstrap percentil B = 1000 para marginales, IC Fisher de pingouin para parciales.

## 5. Ejecución

```bash
make exp_005
```

Tiempo estimado: 50-80 min wall en M3 Pro (Llama 3.2 1B), o ~100-130 min wall si fallback a Pythia 2.8B.

**Pendiente de ejecución por instrucción explícita del usuario.** Espera revisión del diseño y decisión sobre orden:
- ¿Ejecutar exp_005 directamente?
- O ¿ejecutar exp_004c PRIMERO para desambiguar el artefacto detectado en exp_004b, y exp_005 sólo si exp_004c sugiere que merece la pena?

Mi recomendación, documentada en §1 caveat, es **exp_004c primero**. Pero el spec literal del usuario para CASO POSITIVO de exp_004b dice diseñar exp_005, así que aquí está el diseño listo para ejecutar.

## 6. Resultados

`[rellenar tras ejecutar]`

## 7. Análisis estadístico

`[rellenar tras ejecutar]`

## 8. Conclusiones

`[rellenar tras ejecutar; comparar contra exp_004b Pythia 1.4B; verificar si R1 replica el resultado positivo y si el patrón completo (P1/P2/P3/P4/P5 de exp_004b) sigue presente con la misma forma cualitativa]`

## 9. Próximos pasos sugeridos

`[rellenar tras ejecutar]`
