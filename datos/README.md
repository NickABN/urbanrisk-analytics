# Datos

## Origen

Registros de incidentes viales atendidos por el **Centro de Comando, Control, Cómputo, Comunicaciones y Contacto Ciudadano (C5)** del Gobierno de la Ciudad de México.

- **Archivo:** `inViales_2022_2024.csv`
- **Tamaño:** ~104 MB
- **Periodo:** enero 2022 – diciembre 2024
- **Granularidad:** un registro por llamada de emergencia atendida
- **Fuente:** Portal de Datos Abiertos de la Ciudad de México — `datos.cdmx.gob.mx`

El archivo completo **no se versiona en este repositorio** por tamaño y por términos de uso. Debe descargarse del portal de origen.

## Cómo apuntar al archivo completo

```bash
set RUTA_CSV=C:\ruta\a\inViales_2022_2024.csv     # Windows
export RUTA_CSV=/ruta/a/inViales_2022_2024.csv    # Linux / macOS
```

Si la variable no está definida, los módulos usan `datos/muestra/inViales_muestra.csv`.

## La muestra — en preparación

`muestra/inViales_muestra.csv` **todavía no forma parte del repositorio.** Por ahora el flujo requiere el CSV completo, apuntado mediante `RUTA_CSV`.

Cuando esté disponible será un subconjunto estratificado por alcaldía, año y franja horaria, de unos 5 MB, con semilla fija. Su propósito será permitir que el flujo se ejecute de principio a fin sin descargar los 104 MB. **Las cifras que produzca no coincidirán con las del documento del TFM:** servirá para verificar que el código corre y que la lógica es la descrita, no para reproducir los resultados exactos.

## Campos utilizados

| Campo | Tipo | Uso |
|---|---|---|
| `fecha_creacion`, `hora_creacion` | texto | Se combinan en `dt_creacion`, la marca temporal del incidente |
| `fecha_cierre`, `hora_cierre` | texto | Se combinan en `dt_cierre`; su diferencia da el tiempo de atención |
| `codigo_cierre` | texto | **Filtro principal.** Solo se conservan los registros con código `A` (afirmativo: incidente confirmado) |
| `alcaldia_catalogo` | texto | Unidad territorial de análisis |
| `colonia_catalogo` | texto | Análisis exploratorio de concentración |
| `latitud`, `longitud` | numérico | Clustering geográfico con DBSCAN |
| `tipo_incidente_c4`, `incidente_c4` | texto | Tipificación del evento |
| `clas_con_f_alarma` | texto | Clasificación de severidad |
| `tipo_entrada` | texto | Canal de reporte |
| `dia_semana` | texto | Variable calendárica |

## Decisiones de filtrado

**Exclusión de 2021.** El registro contiene 73 filas con fecha anterior a 2022, fuera del periodo declarado del proyecto. Se descartan por coherencia con el entregable 1.

**Filtro por código de cierre `A`.** El C5 clasifica el desenlace de cada llamada. Solo el código `A` corresponde a incidentes confirmados en sitio. Los demás códigos incluyen falsas alarmas, duplicados y llamadas informativas, que introducirían ruido en el conteo.

**Normalización de texto.** Todos los campos categóricos se pasan a mayúsculas sin acentos antes de agrupar, para evitar que variantes ortográficas del mismo valor se cuenten como categorías distintas.

## Advertencia sobre la naturaleza del dato

Este registro recoge **llamadas de emergencia, no siniestros verificados por autoridad de tránsito**. Implica dos sesgos que no están cuantificados y que conviene tener presentes al leer cualquier resultado:

- **Sesgo de reporte.** Zonas con mayor densidad de población o mayor propensión a llamar al 911 aparecen sobrerrepresentadas.
- **Subregistro.** Los incidentes menores que se resuelven entre las partes no generan llamada y no existen en este dato.
