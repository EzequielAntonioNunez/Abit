# Proyecto de investigación de Abit (Identy Labs)

# Information as Transformation

Investigación empírica sobre si una métrica basada en cambio estructural interno del modelo ("sorpresa significativa") predice mejor el impacto cognitivo de un token que la entropía de Shannon, usando LLMs open-source.

## Estado

Pre-experimentación. Fase 0: setup y replicación de baselines.

## Documentación

- `docs/plan.md` — Plan de investigación (hipótesis, fases, criterios)
- `docs/design.md` — Diseño experimental (modelos, datasets, métricas, estadística)
- `docs/experiment_template.md` — Cómo documentar cada experimento
- `docs/decisions/` — Decisiones técnicas (ADRs)
- `CLAUDE.md` — Guía operativa para Claude Code

## Instalación

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
make install
source .venv/bin/activate
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

## Experimentos

```bash
make exp_001
```

Cada experimento vive en `experiments/exp_NNN_descripcion/` y se autodocumenta.

## Estructura

```
src/iat/         librería: modelos, datasets, métricas, stats, io
experiments/     un directorio por experimento, inmutable tras ejecución
docs/            plan, diseño, decisiones, plantilla de experimentos
tests/           tests unitarios
results/         resúmenes globales cross-experiment
data/            datasets cacheados (gitignored)
```

## Reglas mínimas

- Pre-registrar predicción cuantitativa antes de correr cualquier experimento.
- Reportar resultados negativos.
- Seeds fijos, versiones pinneadas.
- Un cambio por experimento.

## Licencia

MIT.
