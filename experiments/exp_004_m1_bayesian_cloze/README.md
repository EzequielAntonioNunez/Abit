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

| # | Predicción | Métrica | Rango / umbral | Lectura |
|---|---|---|---|---|
| P1 | Sanity: M1 correlaciona con Shannon | Spearman ρ(M1, surprisal_bits) | ∈ [0.4, 0.8] | Si ρ < 0.3 hay un bug en M1; si ρ > 0.85 M1 es redundante con Shannon |
| P2 | M1 aporta señal funcional ortogonal | Spearman parcial ρ(M1, delta_cloze \| surprisal_bits) | > 0.15 | Predicción primaria de descubrimiento |
| P3 | Sanity cloze: delta_cloze no es ruido | Spearman ρ(delta_cloze, surprisal_bits) | ≠ 0 con IC excluyendo 0 | Control de calidad de la señal cloze |
| P4 | Robustez: M1 no es subproducto de log_freq | Spearman parcial ρ(M1, surprisal_bits \| log_freq) | > 0.3 | Control análogo al de exp_002b |

**Criterios de falsación**:
- Si P1 falla (ρ < 0.3): hay un bug en `m1_and_cloze_block`. Parar y depurar antes de interpretar P2.
- Si P1 está dentro de rango pero **P2 falla** (ρ_parcial < 0.10 o IC superior < 0.15): M1 es esencialmente una transformación monótona de Shannon. La línea "información ortogonal a Shannon" queda **cerrada como medida intrínseca**: cualquier valor predictivo de M1 ya está en Shannon. exp_005 pivotaría a tareas downstream con datasets externos (Counterfact, LAMBADA) antes de cerrar la línea entera.
- Si P2 se cumple (ρ_parcial > 0.15) **y** P3 confirma señal cloze: H3 del plan recibe el primer apoyo empírico positivo del proyecto. exp_005 escalaría: barrido de k, cross-model con Pythia 2.8B, tareas downstream externas.

**Notas sobre dependencias entre predicciones**: P1 es prerrequisito metodológico (sanity). P2 es la pregunta de fondo. P3 y P4 son controles. Sólo P1 y P2 son condiciones determinantes para "hay descubrimiento o no".

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
