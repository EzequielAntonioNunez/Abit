# ADR 0001 — Stack inicial

**Fecha**: 2026-05-13
**Estado**: aceptada

## Contexto

Necesitamos definir un stack inicial estable para un proyecto de investigación empírica sobre métricas de información en LLMs. El stack debe correr local en M3 Pro 16 GB y escalar a GPUs cloud sin reescritura.

## Decisión

| Categoría | Elección | Razón |
|-----------|----------|-------|
| Lenguaje | Python 3.11 | Ecosistema ML estándar |
| Gestor de paquetes | uv | Velocidad, lockfile reproducible, único entorno |
| ML core | torch + transformers + accelerate | Estándar de facto open-source |
| Backend local | torch MPS | Apple Silicon nativo |
| Datasets | huggingface/datasets | Versionado, caché, splits estandarizados |
| Tracking | mlflow autoalojado | Local-first, sin vendor lock-in |
| Configuración | hydra-core + omegaconf | Composable, override por CLI |
| Estadística | scipy + statsmodels + pingouin | Tests, IC, regresión múltiple |
| Análisis | pandas + pyarrow | Parquet por defecto para records |
| Lint/types | ruff + mypy strict | Calidad sin discusión |
| Tests | pytest + hypothesis | Property-based donde aplique |

## Consecuencias

- Reproducibilidad mejorada por uv lockfile vs. pip freeze.
- Coste cero de vendor lock-in: todas las piezas son open-source.
- Trabajo cross-platform inmediato (M3 local → CUDA cloud) sin cambios de código.
- Decisión de no usar `lightning` o frameworks de entrenamiento: este es un proyecto de inferencia y análisis, no entrenamiento. Añaden complejidad innecesaria.

## Alternativas consideradas

- **JAX**: descartado en fase inicial. Menos compatible con MPS, menor masa crítica de modelos open-source con pesos JAX.
- **vLLM**: descartado en fase inicial. Optimizado para serving, no para extracción de activaciones internas que necesitamos.
- **Poetry**: descartado por velocidad. uv es ~10x más rápido en resolución.
