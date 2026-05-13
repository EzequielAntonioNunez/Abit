# Proyecto de investigación de Abit (Identy Labs)

# CLAUDE.md

Guía operativa para Claude Code en este repositorio. Léeme antes de cualquier acción.

## 1. Qué es este proyecto

Investigación empírica con datos reales. No es un sandbox ni un tutorial. Cada experimento debe ser reproducible, pre-registrado y documentado de forma idéntica.

Lectura obligatoria antes de modificar nada:
- `docs/plan.md`
- `docs/design.md`
- `docs/experiment_template.md`

## 2. Stack

- Python 3.11
- `uv` como gestor de paquetes y entornos virtuales
- `torch` (backend MPS en M3 Pro, CUDA en cloud)
- `transformers`, `datasets`, `accelerate`
- `numpy`, `scipy`, `statsmodels`, `pingouin`
- `hydra-core` para configuración
- `mlflow` para tracking experimental
- `pytest`, `ruff`, `mypy`

Si necesitas una dependencia adicional, la añades a `pyproject.toml` y reinstalas con `uv pip install -e ".[dev]"`. Nunca usar `pip install` directo.

## 3. Convenciones de código

- Sin comentarios obvios. Solo comentar el *porqué*, nunca el *qué*.
- Sin emojis en código, docstrings, commits ni outputs.
- Type hints obligatorios en funciones públicas.
- `from __future__ import annotations` en cada módulo.
- f-strings, nunca `.format()` ni `%`.
- `@torch.no_grad()` en cualquier función de inferencia.
- Shape de tensors documentado en docstring cuando ≥ 2D.
- Funciones puras siempre que sea posible.
- `pathlib.Path`, nunca `os.path`.
- Logging vía `logging` o `mlflow`, no `print()`.

## 4. Workflow para un experimento nuevo

1. Copiar `experiments/_template/` a `experiments/exp_NNN_descripcion_corta/`.
2. NNN = siguiente número con tres dígitos (`exp_002`, `exp_003`...).
3. Completar el README en este orden, antes de ejecutar:
   - Hipótesis específica
   - Predicción cuantitativa pre-registrada
   - Cambio respecto al experimento anterior
4. Editar `config.yaml`.
5. Implementar `run.py`. Reutilizar lo que ya existe en `src/iat/`.
6. Ejecutar.
7. Completar el README con: Resultados, Análisis, Conclusiones, Próximos pasos.
8. Commitear.

**Un experimento no está terminado hasta que su README está completo. Sin excepciones.**

## 5. Documentación de cada experimento

El README de cada experimento sigue este orden exacto. Ver `docs/experiment_template.md`.

1. Código y título (`exp_NNN — Título`)
2. Metadatos (fecha, modelo de Claude Code, autor)
3. Hipótesis
4. Predicción pre-registrada
5. Cambios respecto al experimento anterior
6. Configuración
7. Ejecución
8. Resultados
9. Análisis estadístico
10. Conclusiones
11. Próximos pasos sugeridos

Es **idéntico para todos los experimentos**. No alterar el orden ni los títulos.

## 6. Reproducibilidad

Toda ejecución produce, en `experiments/exp_NNN/results/<timestamp>/`:

- `config.snapshot.yaml` — config exacta usada
- `summary.json` — métricas agregadas y estadísticos
- `records.parquet` — observaciones individuales
- `git_sha.txt` — SHA del commit
- `env.txt` — `uv pip freeze`
- `figures/` — gráficos si los hay

Seed obligatorio en `config.yaml`, propagado a `random`, `numpy.random`, `torch.manual_seed`.
Modelo pinneado por revisión de HuggingFace (no solo nombre).
Dataset pinneado por split y commit hash cuando aplique.

## 7. Naming

| Elemento | Patrón | Ejemplo |
|----------|--------|---------|
| Experimento | `exp_NNN_descripcion_corta` | `exp_001_shannon_baseline` |
| ADR | `NNNN-titulo-corto.md` | `0001-stack-inicial.md` |
| Documento doc | `snake_case.md` | `experiment_template.md` |
| Módulo Python | `snake_case.py` | `bayesian.py` |
| Variable / función | `snake_case` | `compute_surprisal` |
| Clase | `PascalCase` | `LoadedModel` |
| Constante | `UPPER_SNAKE_CASE` | `DEFAULT_SEED` |
| Archivo de resultados | nombre fijo: `summary.json`, `records.parquet` | |

## 8. Comandos

```bash
make install        # crear .venv e instalar
make test           # pytest
make lint           # ruff check + mypy
make format         # ruff format
make exp_001        # ejecutar experimento 001
```

Para un experimento nuevo, añadir target en `Makefile`:
```makefile
exp_NNN:
	python experiments/exp_NNN_descripcion/run.py
```

## 9. Reglas estadísticas

- **Pre-registro obligatorio**: predicción cuantitativa en el README antes de ejecutar.
- Bootstrap (B ≥ 1000) para intervalos de confianza.
- Corrección por comparaciones múltiples cuando aplique (Bonferroni o BH-FDR).
- Reportar siempre: tamaño de muestra, seed, modelo, dataset, hardware.
- Resultados negativos se reportan completos. Nunca se silencian.

## 10. Anti-patterns prohibidos

- Comentarios decorativos o docstrings vacíos.
- Emojis en cualquier archivo.
- Commitear `data/`, modelos descargados, resultados raw > 1 MB.
- Correr experimentos sin pre-registro.
- Modificar o renombrar experimentos previos: son histórico inmutable.
- Silenciar warnings sin justificación documentada.
- `print()` para tracking de runs.
- Instalar paquetes con pip directo, fuera de `pyproject.toml`.
- "Optimizar" el código de un experimento ya ejecutado.

## 11. Idioma

- Documentación, docstrings, mensajes de commit: **español**.
- Nombres de archivos, variables, funciones, clases: **inglés**.

## 12. Cuando dudes

- Métrica nueva → módulo en `src/iat/metrics/`, no en `run.py`.
- Dataset nuevo → loader en `src/iat/datasets.py`.
- Decisión metodológica con consecuencias → ADR en `docs/decisions/`.
- Cambio al plan → añadir apartado fechado en `docs/plan.md`, **no reescribir el pasado**.
- Resultado inesperado → documentarlo y abrir un experimento de verificación, no asumir.

## 13. Hardware y escala

- Desarrollo local: Mac M3 Pro 16 GB, backend MPS.
- Modelos ≤ 3B viables localmente.
- Modelos ≥ 7B requieren cloud (Modal, Runpod). Documentar coste estimado antes de lanzar.
- Si un experimento requiere > 4h en local, considerar paralelización o reducir N.

## 14. Política de commits

- Un commit por cambio lógico.
- Mensaje en español, imperativo: "añadir métrica de activation drift".
- Nunca commitear con tests rojos.
- Nunca commitear con `mypy` o `ruff` rojos.
- El estado `main` siempre debe poder correr `make exp_001`.
