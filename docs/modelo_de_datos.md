# Modelo de datos

El prototipo organiza la información en un **esquema en estrella**: una tabla de hechos con la medida de interés y cuatro dimensiones que la describen. Es la estructura habitual en analítica porque permite agregar por cualquier combinación de dimensiones sin recorrer uniones complejas, y porque Power BI la consume de forma nativa.

## Diagrama

```
                      ┌──────────────────────┐
                      │   dim_alcaldia       │
                      │──────────────────────│
                      │ PK id_alcaldia       │
                      │    nombre            │
                      │    poblacion         │
                      │    superficie_km2    │
                      │    cluster_kmeans    │
                      └──────────┬───────────┘
                                 │
┌────────────────────┐           │           ┌────────────────────┐
│    dim_tiempo      │           │           │     dim_franja     │
│────────────────────│           │           │────────────────────│
│ PK id_fecha        │           │           │ PK id_franja       │
│    fecha           │           │           │    etiqueta        │
│    anio            │           │           │    hora_inicio     │
│    mes             │           │           │    hora_fin        │
│    trimestre       │           │           └─────────┬──────────┘
│    dia_semana      │           │                     │
│    es_finde        │           │                     │
│    quincena        │           │                     │
│    dia_del_anio    │           │                     │
└─────────┬──────────┘           │                     │
          │                      │                     │
          │        ┌─────────────┴─────────────────────┴──┐
          └────────┤          hecho_riesgo                │
                   │──────────────────────────────────────│
                   │ PK (id_alcaldia, id_fecha, id_franja)│
                   │    n_incidentes          [medida]    │
                   │    nivel_riesgo          [derivada]  │
                   │    prob_bajo                         │
                   │    prob_medio                        │
                   │    prob_alto                         │
                   │    indice_riesgo                     │
                   │    lag_1d … lag_franja   [10 vars]   │
                   └──────────────────┬───────────────────┘
                                      │
                          ┌───────────┴────────────┐
                          │   dim_tipo_incidente   │
                          │────────────────────────│
                          │ PK id_tipo             │
                          │    tipo_incidente_c4   │
                          │    clasificacion_alarma│
                          │    es_grave            │
                          └────────────────────────┘
```

La relación con `dim_tipo_incidente` es de grano más fino que la tabla de hechos principal y se materializa en una tabla puente (`hecho_incidente`) que conserva el registro individual. La tabla de hechos agregada es la que alimenta el modelo y el tablero.

---

## `hecho_riesgo` — tabla de hechos

Grano: **una fila por alcaldía, día y franja horaria**. 50 560 filas en el periodo completo.

| Campo | Tipo | Dominio | Nulos | Origen |
|---|---|---|---|---|
| `id_alcaldia` | entero | 1–16 | No | `dim_alcaldia` |
| `id_fecha` | entero | AAAAMMDD | No | `dim_tiempo` |
| `id_franja` | entero | 1–4 | No | `dim_franja` |
| `n_incidentes` | entero | 0 – 87 | No | Conteo agregado. **18,57 % de las filas valen cero** |
| `nivel_riesgo` | texto | Bajo / Medio / Alto | No | Derivada: terciles de `n_incidentes` calculados sobre entrenamiento |
| `prob_bajo` | decimal | 0 – 1 | No | Salida del modelo |
| `prob_medio` | decimal | 0 – 1 | No | Salida del modelo |
| `prob_alto` | decimal | 0 – 1 | No | Salida del modelo |
| `indice_riesgo` | decimal | 0 – 100 | No | Derivada: combinación ponderada de las tres probabilidades |
| `lag_1d`, `lag_2d`, `lag_7d`, `lag_14d` | decimal | ≥ 0 | Sí, al inicio de cada serie | Conteo de la misma celda 1, 2, 7 y 14 días antes |
| `mm_7d`, `mm_28d` | decimal | ≥ 0 | Sí, al inicio | Media móvil sobre el valor desplazado |
| `ds_28d` | decimal | ≥ 0 | Sí, al inicio | Desviación típica móvil |
| `media_hist` | decimal | ≥ 0 | Sí, al inicio | Media acumulada de todo el pasado disponible |
| `total_dia_prev` | decimal | ≥ 0 | Sí, primer día | Total de la alcaldía el día anterior, todas las franjas |
| `lag_franja` | decimal | ≥ 0 | Sí, primera franja | Conteo de la franja inmediatamente anterior |

Las filas sin historia suficiente se descartan en vez de imputarse. Imputar un retardo equivale a inventar pasado.

---

## `dim_alcaldia`

| Campo | Tipo | Notas |
|---|---|---|
| `id_alcaldia` | entero | Clave |
| `nombre` | texto | Normalizado a mayúsculas sin acentos |
| `poblacion` | entero | Censo 2020, INEGI. Permite normalizar por habitante |
| `superficie_km2` | decimal | Permite calcular densidad de incidentes |
| `cluster_kmeans` | entero | Grupo asignado por el modelo descriptivo |

---

## `dim_tiempo`

| Campo | Tipo | Notas |
|---|---|---|
| `id_fecha` | entero | AAAAMMDD |
| `fecha` | fecha | |
| `anio`, `mes`, `trimestre` | entero | |
| `dia_semana` | texto | Categórica ordenada, lunes a domingo |
| `es_finde` | booleano | Sábado o domingo |
| `quincena` | booleano | Día ≥ 15. Captura el efecto de los ciclos de pago |
| `dia_del_anio` | entero | 1–366 |

---

## `dim_franja`

| `id_franja` | `etiqueta` | `hora_inicio` | `hora_fin` |
|---|---|---|---|
| 1 | Madrugada (00-05) | 0 | 5 |
| 2 | Mañana (06-11) | 6 | 11 |
| 3 | Tarde (12-17) | 12 | 17 |
| 4 | Noche (18-23) | 18 | 23 |

Los bloques de seis horas no son arbitrarios: corresponden a los turnos operativos de los cuerpos de emergencia, de modo que una predicción por franja es directamente accionable para asignar recursos.

---

## `dim_tipo_incidente`

| Campo | Tipo | Notas |
|---|---|---|
| `id_tipo` | entero | Clave |
| `tipo_incidente_c4` | texto | Catálogo del C5 |
| `clasificacion_alarma` | texto | Clasificación de severidad, 4 categorías |
| `es_grave` | booleano | Derivada, usada por el modelo de severidad |

---

## Capa física

En el prototipo el almacenamiento es **Parquet** sobre el sistema de archivos, con particionado lógico por etapa:

| Archivo | Contenido | Tamaño aproximado |
|---|---|---|
| `datos/intermedios/datos_filtrado.parquet` | Registro individual tras el filtrado | ~40 MB |
| `datos/intermedios/panel_riesgo.parquet` | Tabla de hechos sin retardos | ~2 MB |
| `datos/intermedios/panel_lags.parquet` | Tabla de hechos completa | ~6 MB |
| `tablero/rejilla_riesgo.csv` | Rejilla de predicción para el tablero | ~800 KB |

Parquet es la elección correcta para este volumen: columnar, comprimido, con tipos preservados y lectura parcial por columnas. Un despliegue en producción sustituiría esta capa por una base analítica —PostgreSQL con particionado por fecha, o BigQuery— sin que cambie el esquema lógico descrito arriba.

## Por qué la rejilla de predicción es un CSV y no una consulta

El tablero necesita responder «¿qué riesgo hay en Iztapalapa, un viernes de marzo por la noche?» sin invocar Python. La rejilla precalcula las 10 752 combinaciones posibles de la dimensión estructural y las guarda en una tabla de 803 KB que Power BI carga entera en memoria.

Es una decisión de arquitectura con una consecuencia explícita: solo las variables estructurales pueden entrar, porque son las únicas conocidas de antemano. Los retardos quedan fuera. El detalle está en [`decisiones.md`](decisiones.md).
