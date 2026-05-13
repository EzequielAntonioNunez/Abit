# Proyecto de investigación de Abit (Identy Labs)

# exp_NNN — Título corto del experimento

> Copiar este directorio a `experiments/exp_NNN_descripcion_corta/`, asignar el siguiente número, completar todas las secciones marcadas como `[rellenar...]` ANTES de ejecutar (salvo Resultados, Análisis, Conclusiones y Próximos pasos, que se completan tras ejecutar).

## Metadatos

- **Fecha de diseño**: `[YYYY-MM-DD]`
- **Fecha de ejecución**: `[YYYY-MM-DD]` (rellenar tras correr)
- **Autor**: `[nombre]`
- **Claude Code model**: `[claude-...]`
- **Hardware**: `[M3 Pro local | Modal H100 | etc.]`

## 1. Hipótesis

`[rellenar: hipótesis específica de este experimento, en una frase. Debe ser falsable.]`

## 2. Predicción pre-registrada

`[rellenar: predicción cuantitativa con rangos numéricos esperados. Ej: "Spearman rho ∈ [0.3, 0.6]". Si los datos quedan fuera de ese rango, hay que explicar por qué.]`

**Criterio de falsación**: `[rellenar: qué resultado concreto falsea la hipótesis.]`

## 3. Cambios respecto al experimento anterior

`[rellenar: qué se modifica respecto a exp_NNN-1. Un cambio por experimento. Si es el primero, "ninguno: baseline inicial".]`

## 4. Configuración

Ver `config.yaml` para parámetros completos.

Decisiones clave de este experimento:
- `[rellenar: modelo, dataset, N, hiperparámetros relevantes]`

## 5. Ejecución

```bash
python experiments/exp_NNN_descripcion/run.py
```

Tiempo estimado: `[rellenar]`

## 6. Resultados

`[rellenar tras ejecutar. Tablas, métricas agregadas. Referenciar archivos en results/<timestamp>/]`

Última ejecución: `[ruta a results/<timestamp>/]`

## 7. Análisis estadístico

`[rellenar: tests realizados, intervalos de confianza, correcciones por comparaciones múltiples.]`

## 8. Conclusiones

`[rellenar: ¿se confirma la hipótesis? ¿qué porción del intervalo predicho se cumplió? Honestidad obligatoria con resultados negativos.]`

## 9. Próximos pasos sugeridos

`[rellenar: 1-3 experimentos siguientes derivados de este resultado, con justificación.]`
