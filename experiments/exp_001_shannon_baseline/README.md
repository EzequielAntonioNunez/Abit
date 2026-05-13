# Proyecto de investigación de Abit (Identy Labs)

# exp_001 — Shannon Baseline

## Metadatos

- **Fecha de diseño**: 2026-05-13
- **Fecha de ejecución**: `[pendiente]`
- **Autor**: Ezequiel
- **Claude Code model**: `[rellenar tras ejecutar]`
- **Hardware**: M3 Pro local

## 1. Hipótesis

Es posible reproducir el surprisal Shannon token-a-token en Gemma 2 2B sobre WikiText-103 de forma estable y obtener una perplejidad coherente con valores publicados para este orden de magnitud de modelo.

Este experimento **no testea la teoría central** del proyecto. Su único objetivo es validar que el pipeline (carga de modelo, tokenización, forward pass, cálculo de NLL) funciona y produce números razonables. Es prerrequisito de todos los experimentos posteriores.

## 2. Predicción pre-registrada

- **Perplejidad de Gemma 2 2B sobre WikiText-103 validation**: entre 8 y 20.
- **Surprisal medio por token**: entre 3 y 4.3 bits.
- **Surprisal p95**: > 8 bits (cola larga esperada por nombres propios y términos raros).
- **Tiempo de ejecución en M3 Pro**: < 20 min para 100 docs y contextos de 256 tokens.

**Criterio de falsación**: si la perplejidad sale fuera del rango [5, 50], hay un bug. Si todos los tokens tienen surprisal entre 0 y 1, hay un bug. Investigar antes de continuar.

## 3. Cambios respecto al experimento anterior

Ninguno. Es el baseline inicial.

## 4. Configuración

Ver `config.yaml`.

Decisiones clave:
- Modelo: `google/gemma-2-2b` en bfloat16 (encaja en M3 Pro 16 GB con holgura).
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

`[rellenar tras ejecutar]`

Última ejecución: `[results/<timestamp>/summary.json]`

## 7. Análisis estadístico

`[rellenar: distribución del surprisal, perplejidad, comparación con valores esperados]`

## 8. Conclusiones

`[rellenar]`

## 9. Próximos pasos sugeridos

Si los baseline son correctos:
- **exp_002**: añadir activation drift en última capa, medir correlación con Shannon en los mismos tokens. Predicción: Spearman ρ ∈ [0.3, 0.6].

Si los baselines son anómalos:
- **exp_001b**: reproducir con Pythia 1.4B para descartar bug específico de Gemma.
