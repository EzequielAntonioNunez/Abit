# Proyecto de investigación de Abit (Identy Labs)

# exp_001 — Shannon Baseline

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: 2026-05-13 (16:10:05 → 16:23:25 hora local, UTC 14:10:05Z)
- **Autor**: Ezequiel
- **Claude Code model**: claude-opus-4-7 (1M context)
- **Hardware**: M3 Pro local, backend `mps`, dtype `bfloat16`

## 1. Hipótesis

Es posible reproducir el surprisal Shannon token-a-token en Pythia 1.4B sobre WikiText-103 de forma estable y obtener una perplejidad coherente con valores publicados para este orden de magnitud de modelo.

Este experimento **no testea la teoría central** del proyecto. Su único objetivo es validar que el pipeline (carga de modelo, tokenización, forward pass, cálculo de NLL) funciona y produce números razonables. Es prerrequisito de todos los experimentos posteriores.

**Sustitución de modelo (pre-ejecución, 2026-05-13)**: el modelo originalmente preregistrado era `google/gemma-2-2b`, pero requiere autenticación HuggingFace y la sesión actual no dispone de credenciales (`OSError: gated repo` al cargar el tokenizer). Se sustituye por `EleutherAI/pythia-1.4b` (open access, escala comparable, ya contemplado en el diseño como sujeto experimental). La sustitución se aplica antes de ejecutar, por lo que las secciones 1-5 reflejan el modelo efectivo. La sustitución no altera la naturaleza del experimento (validar pipeline + sanity check de perplejidad).

## 2. Predicción pre-registrada

- **Perplejidad de Pythia 1.4B sobre WikiText-103 validation**: entre 8 y 20.
- **Surprisal medio por token**: entre 3 y 4.3 bits.
- **Surprisal p95**: > 8 bits (cola larga esperada por nombres propios y términos raros).
- **Tiempo de ejecución en M3 Pro**: < 20 min para 100 docs y contextos de 256 tokens.

**Criterio de falsación**: si la perplejidad sale fuera del rango [5, 50], hay un bug. Si todos los tokens tienen surprisal entre 0 y 1, hay un bug. Investigar antes de continuar.

## 3. Cambios respecto al experimento anterior

Ninguno respecto al diseño base. Único cambio frente al pre-registro original: sustitución del modelo `google/gemma-2-2b` por `EleutherAI/pythia-1.4b` por falta de credenciales HuggingFace en la sesión de ejecución (documentado en sección 1).

## 4. Configuración

Ver `config.yaml`.

Decisiones clave:
- Modelo: `EleutherAI/pythia-1.4b` en bfloat16 (sustitución, ver sección 1). Pythia 1.4B encaja en M3 Pro 16 GB con holgura y es no gated.
- Dataset: WikiText-103 split `validation` (filtrado por longitud mínima 200 chars, máx 100 docs).
- Contextos truncados a 256 tokens.
- Tokens de warmup (primeros 50) excluidos del análisis para evitar sesgo de inicio de documento.
- Seed: 42.

## 5. Ejecución

```bash
make exp_001
```

Tiempo estimado: 5-15 min en M3 Pro.

## 6. Resultados

Run de referencia: `results/20260513T141005Z/`

| Métrica | Valor |
|---|---|
| Modelo efectivo | `EleutherAI/pythia-1.4b` |
| Device / dtype | `mps` / `torch.bfloat16` |
| Documentos usados | 98 (de 100 filtrados; 2 descartados por longitud < 60 tokens tras tokenizar) |
| Tokens evaluados (N) | 9 955 |
| Perplejidad | **12.459** |
| Surprisal medio | **3.639 bits/token** |
| Surprisal mediano | **2.306 bits/token** |
| Surprisal p95 | **11.564 bits/token** |
| Tiempo real de ejecución | **~13 min 20 s** (16:10:05 → 16:23:25 local; download de pesos + dataset incluido) |
| Tiempo de forward pass solo | ~15 s para 100 docs en M3 Pro |
| SHA git en el run | `d6f0dde` |

Artefactos en el run: `config.snapshot.yaml`, `summary.json`, `records.parquet` (110 KB), `git_sha.txt`, `env.txt`.

## 7. Análisis estadístico

Comparación punto a punto contra el pre-registro (sección 2):

| Predicción | Rango registrado | Valor observado | ¿Se cumple? |
|---|---|---|---|
| Perplejidad ∈ [8, 20] | [8, 20] | 12.459 | Sí |
| Surprisal medio ∈ [3, 4.3] bits | [3.0, 4.3] | 3.639 | Sí |
| Surprisal p95 > 8 bits | > 8 | 11.564 | Sí |
| Tiempo en M3 Pro < 20 min | < 20 min | ~13 min 20 s | Sí |
| Criterio de falsación: perplejidad ∈ [5, 50] | [5, 50] | 12.459 | Sí (no se dispara) |
| Criterio de falsación: distribución no colapsada | rango (0, 1) en todos los tokens | mediana 2.31, p95 11.56, dispersión amplia | No se dispara |

La relación `perplejidad = 2^avg_surprisal_bits` se verifica numéricamente: `2^3.6391 ≈ 12.459`. La identidad cierra dentro de la precisión esperada, lo que es un sanity check adicional de que el cálculo bits ↔ perplejidad es consistente.

La asimetría entre mediana (2.31) y media (3.64) es coherente con la cola larga prevista: gran parte de los tokens son altamente predecibles (función gramatical, BPE-pieces, tokens frecuentes) y un subconjunto pequeño aporta la mayor parte de la NLL. El p95 a 11.56 bits confirma la presencia de tokens raros / nombres propios, en línea con la predicción cualitativa.

Tests estadísticos formales (bootstrap de IC, etc.) **no se realizan en este experimento**: por diseño es un baseline de pipeline, no un contraste de hipótesis con grupo de comparación. Se registran únicamente los estadísticos descriptivos del corpus medido.

## 8. Conclusiones

El pipeline funciona y los baselines son coherentes. Las cuatro predicciones cuantitativas pre-registradas se cumplen y los dos criterios de falsación no se disparan. No se observan anomalías: el modelo carga en `mps` en bfloat16 sin errores numéricos visibles, la NLL es estable y la distribución de surprisal por token tiene la forma esperada (mediana baja, media mayor, cola larga).

Salvedades honestas:
- El modelo efectivo es Pythia 1.4B y no Gemma 2 2B; la sustitución se hizo **antes** de ejecutar y se justificó por falta de credenciales HuggingFace en la sesión (sección 1). Esto no invalida el experimento como prueba de pipeline, pero implica que los valores numéricos exactos no son directamente comparables con la literatura publicada para Gemma 2 2B; los rangos pre-registrados estaban deliberadamente abiertos para cubrir ambas escalas.
- La perplejidad 12.46 es ligeramente más baja que el ~14-16 que suele reportarse para Pythia 1.4B sobre WikiText-103 validation completa. La diferencia es esperable porque: (i) se evalúan sólo 98 docs, (ii) los contextos están truncados a 256 tokens, (iii) los 50 primeros tokens por documento se excluyen del cómputo (suelen ser los de mayor NLL, lo que baja la media). No constituye una anomalía que requiera depuración.
- Una advertencia de `transformers` indica que `output_hidden_states=True` es ignorado en `generation_config`; el flag sólo es relevante para llamadas a `.generate()` y no afecta a la llamada directa a `model(...)` usada aquí. No tiene impacto sobre el cálculo de surprisal en este experimento, pero conviene tenerlo en cuenta cuando los hidden states se necesiten en exp_002.

Veredicto: pipeline validado, listo para usar como infraestructura de las métricas alternativas (M1-M5 del diseño).

## 9. Próximos pasos sugeridos

Dado que los baselines son correctos, se propone proceder con la rama "feliz":

- **exp_002 — Activation drift, última capa**: añadir un segundo forward pass por documento sin el token objetivo (o un forward de `C` y otro de `C, t_i`) y calcular la métrica M2_last definida en `docs/design.md` §3. Reusar el sampling y el split de exp_001 sobre los mismos tokens para que ambas métricas estén alineadas por token. Predicción pre-registrada (ya enunciada): Spearman ρ(surprisal_bits, M2_last) ∈ [0.3, 0.6]. Falsadores: ρ > 0.9 (M2 colapsa a Shannon, H3 trivializada) o ρ < 0 (señal opuesta, requiere depuración antes de seguir).
- **exp_003 (condicional)** — si exp_002 sale dentro del rango: extender a M2_mean y M2_max para evaluar capa-por-capa. Mantener todo lo demás constante.

Reserva: si exp_002 expone un bug en la cadena de hidden states (la advertencia del logger en exp_001 sugiere vigilar este punto), abrir **exp_002b** con la corrección. No se reabre exp_001.
