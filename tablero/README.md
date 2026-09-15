# Tablero de inteligencia territorial

**Pendiente de construir.** Ver [`../PENDIENTE.md`](../PENDIENTE.md), punto 5.

## Cómo funciona

El tablero no invoca ningún modelo. Consulta `rejilla_riesgo.csv`, una tabla de 10 752 filas que enumera de antemano **todas** las combinaciones posibles de alcaldía × franja × día de la semana × mes × quincena, cada una con su nivel de riesgo predicho y sus tres probabilidades.

Esto convierte la inferencia en una búsqueda. El tablero funciona sin servidor, sin latencia y sin dependencias de Python.

El coste de la decisión está documentado en [`../docs/decisiones.md`](../docs/decisiones.md), entrada D-07: el motor es el modelo estructural, catorce milésimas de F1 por debajo del mejor modelo, porque es el único precalculable.

## Generar la rejilla

```bash
python -m src.tablero
```

Deja `rejilla_riesgo.csv` en esta carpeta, codificado en UTF-8 con BOM para que Power BI lea bien los acentos.

## Vistas previstas

| Vista | Contenido |
|---|---|
| Mapa | Coropletas por alcaldía, coloreadas por índice de riesgo |
| Semáforo | Nivel Bajo / Medio / Alto de la selección actual |
| Filtros | Día de la semana, franja horaria, mes, quincena |
| Serie temporal | Evolución histórica del conteo de la alcaldía seleccionada |
| Ranking | Alcaldías ordenadas por índice de riesgo en la combinación elegida |

## Campos de `rejilla_riesgo.csv`

| Campo | Descripción |
|---|---|
| `alcaldia_catalogo` | Alcaldía |
| `franja` | Franja horaria de seis horas |
| `dia_semana` | Lunes a domingo |
| `mes`, `trimestre`, `quincena`, `finde` | Variables calendáricas |
| `prob_bajo`, `prob_medio`, `prob_alto` | Probabilidad de cada nivel |
| `nivel_riesgo` | Nivel predicho |
| `nivel_codigo` | 0, 1, 2 |
| `indice_riesgo` | Índice continuo 0–100, para el semáforo |
