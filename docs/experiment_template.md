# Proyecto de investigación de Abit (Identy Labs)

# Cómo documentar un experimento

Este documento define el contrato de documentación. Es **vinculante**.

## Principio

Cada experimento debe poder leerse de forma autónoma. Un lector debe poder reconstruir qué se hizo, por qué, qué se predijo, qué se obtuvo y qué falta, sin abrir el código.

## Estructura obligatoria del `README.md` del experimento

Las secciones deben aparecer en este orden exacto y con estos títulos exactos:

1. **Título**: `# exp_NNN — Título corto`
2. **Metadatos**: fecha de diseño, fecha de ejecución, autor, modelo de Claude Code, hardware.
3. **Hipótesis**: una frase falsable. Si no se puede falsear, no es hipótesis.
4. **Predicción pre-registrada**: rangos numéricos. Si la predicción es "verá algo interesante", no es pre-registro.
5. **Cambios respecto al experimento anterior**: un cambio. Si son varios, dividir en varios experimentos.
6. **Configuración**: link a `config.yaml` y decisiones clave en prosa.
7. **Ejecución**: comando, tiempo estimado.
8. **Resultados**: tablas, números agregados, referencias a archivos en `results/<timestamp>/`.
9. **Análisis estadístico**: tests realizados, IC, correcciones.
10. **Conclusiones**: ¿se confirma la hipótesis? Honestidad con resultados negativos.
11. **Próximos pasos sugeridos**: 1-3 experimentos siguientes con justificación.

## Reglas

- Las secciones 1-7 se completan **antes** de ejecutar.
- Las secciones 8-11 se completan **después** de ejecutar.
- Si después de ejecutar hay que cambiar la hipótesis o la predicción, **no se modifica el documento original**. Se abre un experimento nuevo con la hipótesis revisada.
- Las predicciones cuantitativas son obligatorias. Frases como "esperamos resultados interesantes" no cuentan.
- Si el resultado falsifica la hipótesis, se documenta igual y se marca explícitamente.

## Naming

- Directorio: `exp_NNN_descripcion_corta_snake_case`
- NNN: tres dígitos, zero-padded, **monotónicamente creciente** y nunca reutilizado.
- Si un experimento se aborta a mitad: documentarlo en su README y reservar el número (no eliminar).

## Artefactos de cada run

Cada ejecución crea `results/<timestamp>/` con:

| Archivo | Obligatorio | Contenido |
|---------|-------------|-----------|
| `config.snapshot.yaml` | Sí | Copia exacta del `config.yaml` usado |
| `summary.json` | Sí | Métricas agregadas: medias, medianas, IC, tamaños de muestra |
| `records.parquet` | Recomendado | Observaciones individuales por token |
| `git_sha.txt` | Sí | SHA del commit |
| `env.txt` | Sí | Output de `uv pip freeze` |
| `figures/` | Opcional | Gráficos PNG/PDF generados |

Los outputs se generan automáticamente con `iat.io.snapshot_run(...)` y helpers asociados.

## Política de modificación

- Un experimento ya ejecutado es **histórico inmutable**. No se renombra. No se reorganizan archivos.
- Si la implementación tenía un bug grave que invalida resultados: abrir `exp_NNNb` (sufijo `b`, `c`...) con la corrección. Documentar la razón.
- El número original no se reutiliza nunca.
