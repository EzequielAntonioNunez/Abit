# ADR 0002 — Modelo base de Fase 0: Pythia 1.4B

**Fecha**: 2026-05-13
**Estado**: aceptada

## Contexto

El pre-registro original del proyecto (ver `docs/design.md` §1) listaba `google/gemma-2-2b` como sujeto experimental principal para Fase 0 (replicación de baselines y validación del pipeline). En la primera ejecución real (exp_001) se encontró que Gemma 2 2B es un repositorio *gated* en HuggingFace Hub: requiere aceptación manual de licencia y autenticación con token, incluso para descargar el tokenizer. La sesión de ejecución no disponía de credenciales y la carga falló con `OSError: gated repo`.

Esto introduce fricción reproducible:

- Toda persona que ejecute el pipeline desde cero necesita registrarse, aceptar la licencia y configurar `HF_TOKEN` antes de poder correr el experimento más simple.
- Cualquier entorno automatizado (CI, runners en cloud, contenedores) requiere inyección de secretos para algo que debería ser un sanity check.
- El modelo no expone checkpoints intermedios públicos, lo que limita análisis evolutivos previstos en Fase 2.

Como sustitución se usó `EleutherAI/pythia-1.4b` (open weights, sin gating, escala comparable, ya contemplada en `docs/design.md` §1 como uno de los modelos del estudio). El exp_001 con Pythia 1.4B sobre WikiText-103 validation devolvió perplejidad 12.459 y surprisal medio 3.639 bits, ambos dentro de los rangos pre-registrados.

## Decisión

Se consolida `EleutherAI/pythia-1.4b` como **modelo base de Fase 0** y de los experimentos exploratorios e iterativos de Fases 1, 2 y 3. Gemma 2 2B se desplaza a Fase 4 (replicación cross-architecture), momento en que el coste de gestionar credenciales se amortiza contra el valor de comparar arquitecturas distintas.

| Categoría | Elección | Razón |
|-----------|----------|-------|
| Modelo base Fase 0-3 | `EleutherAI/pythia-1.4b` | Open weights, no gated, escala 1.4B comparable, checkpoints intermedios disponibles |
| Modelo de replicación Fase 4 | `google/gemma-2-2b` | Arquitectura distinta; requiere `HF_TOKEN` documentado en setup |
| Modelos secundarios (sin cambio) | Pythia 2.8B, Llama 3.2 1B, Llama 3.2 3B | Según `docs/design.md` §1 |

## Consecuencias

- Reproducibilidad mejorada: cualquiera puede clonar el repo y correr `make exp_001` sin gestionar tokens.
- Se desbloquea el uso de checkpoints intermedios de Pythia (`step1000`, `step10000`, etc.) para análisis evolutivos, que es una ventaja real frente a Gemma para nuestras preguntas (ver `docs/plan.md` §3 pregunta 4).
- Los rangos numéricos del pre-registro (`docs/design.md` §7) se mantienen: están definidos en términos de correlaciones y varianzas residuales, no en valores absolutos de perplejidad atados a un modelo concreto.
- Coste: la comparación cross-architecture con Gemma queda pospuesta a Fase 4 y depende de que la replicación con Pythia 2.8B + Llama 3.2 produzca resultados sólidos en Fase 1-3.
- exp_001 ya ejecutado se mantiene como histórico inmutable; no se reabre.

## Alternativas consideradas

- **Gestionar `HF_TOKEN` y mantener Gemma como base**: descartado. Añade fricción de setup, no aporta nada teórico en Fase 0 (cuyo objetivo es validar pipeline, no medir un modelo específico), y desplaza el problema a CI/cloud en el futuro.
- **Migrar a Llama 3.2 1B como base**: viable, pero Llama también está sujeto a licencia con aceptación. Mismo problema, distinto vendor.
- **Mantener el plan original y bloquear Fase 0 hasta tener token**: descartado. La sesión actual ya validó el pipeline con un modelo open, no hay razón científica para repetir con Gemma sólo para honrar el pre-registro literal.
