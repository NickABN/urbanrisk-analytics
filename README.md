# UrbanRisk Analytics

**Detección de patrones y predicción de accidentes viales en la Ciudad de México mediante Big Data y aprendizaje automático**

Prototipo del Trabajo Final de Maestría. Equipo 2D — Universidad Internacional de La Rioja (UNIR).

---

## Qué hace este proyecto

A partir de los registros de incidentes viales del C5 de la Ciudad de México (2022–2024), este prototipo:

1. Reconstruye una rejilla completa **alcaldía × día × franja horaria** a partir de un registro que solo contiene eventos.
2. Caracteriza los patrones espaciotemporales del riesgo vial mediante análisis exploratorio y clustering geográfico.
3. Clasifica cada celda de la rejilla en tres **niveles de riesgo** con validación cronológica estricta.
4. Genera una **rejilla de predicción precalculada** que alimenta un tablero de inteligencia territorial.

## El resultado principal, y es incómodo

El techo predictivo alcanzado (**F1 macropromediado ≈ 0,67**) resultó ser **estructural, no algorítmico**.

Añadir diez variables de retardo temporal —la hipótesis de partida del trabajo— mejoró el F1 de 0,657 a 0,671: catorce milésimas, un 2,11 % relativo. El análisis de ablación explica por qué: la información estructural (dónde y cuándo) y la dinámica (qué pasó ayer) son en gran medida **redundantes entre sí**.

| Configuración | F1 macro |
|---|---|
| Conjunto completo (zona + calendario + retardos) | 0,6666 |
| Solo variables estructurales (zona + calendario) | 0,6576 |
| Solo variables dinámicas (retardos) | 0,6565 |
| Zona + retardos, sin calendario | 0,6592 |
| Calendario + retardos, sin zona | 0,6582 |

Un oráculo que conociera la media histórica exacta de cada celda —información que ningún modelo real tiene— alcanzaría un F1 de 0,5921. Ese es el orden de magnitud del margen disponible.

A esto se suman una razón varianza/media de 3,44 sobre la rejilla de modelado —muy por encima del valor unitario que exigiría un proceso de Poisson— y asociaciones débiles en todas las variables categóricas (V de Cramér < 0,11). El componente aleatorio del fenómeno domina sobre el sistemático. Más modelo no arregla eso. Se reporta tal cual.

> Sobre la sobredispersión: el documento del TFM citaba 250,78, cifra calculada sobre una agregación por día de la semana y no sobre la rejilla que se modela. Ambas rechazan la equidispersión, pero solo la segunda es pertinente para la decisión de modelado. El razonamiento completo está en [`docs/decisiones.md`](docs/decisiones.md), entrada D-11. Lo detectó la reproducción del flujo, no una revisión del texto.

---

## Reproducir los resultados

### Requisitos

Python 3.11 o superior.

```bash
git clone https://github.com/NickABN/urbanrisk-analytics.git
cd urbanrisk-analytics
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### Los datos

El conjunto completo (~104 MB) **no se versiona en este repositorio**. Procede del portal de datos abiertos de la Ciudad de México y su origen está documentado en [`datos/README.md`](datos/README.md).

**El flujo requiere el CSV completo.** Una muestra reducida y versionada, que permitiría ejecutarlo sin descargar los 104 MB, está en preparación y aún no forma parte del repositorio.

Coloca el CSV y apunta la variable de entorno:

```bash
set RUTA_CSV=C:\ruta\a\inViales_2022_2024.csv     # Windows
export RUTA_CSV=/ruta/a/inViales_2022_2024.csv    # Linux / macOS
```

Si no se define, los módulos buscan `datos/muestra/inViales_muestra.csv`, que es donde vivirá la muestra cuando esté disponible.

### Ejecución

Flujo completo desde la línea de órdenes:

```bash
python -m src.pipeline --todo
```

O por etapas, en este orden:

```bash
python -m src.preparacion       # limpieza y rejilla alcaldía × día × franja
python -m src.caracteristicas   # variables estructurales y de retardo
python -m src.modelos           # entrenamiento con validación cronológica
python -m src.evaluacion        # métricas, ablación e interpretabilidad
python -m src.tablero           # rejilla de predicción precalculada
```

Cada etapa escribe su resultado en `resultados/` y deja un `.json` con las cifras que consume la siguiente. El cuaderno maestro recorre el mismo flujo con narrativa:

```bash
jupyter lab notebooks/UrbanRisk_Analytics.ipynb
```

---

## Organización

```
urbanrisk-analytics/
├── datos/
│   ├── muestra/            muestra reproducible versionada
│   └── README.md           origen del dato completo
├── src/
│   ├── config.py           rutas, constantes y listas de variables
│   ├── carga.py            ingesta y validación de esquema
│   ├── preparacion.py      limpieza y rejilla alcaldía × día × franja
│   ├── caracteristicas.py  ingeniería de variables de retardo
│   ├── modelos.py          entrenamiento y validación cronológica
│   ├── evaluacion.py       métricas, ablación e interpretabilidad
│   ├── tablero.py          rejilla de predicción precalculada
│   └── pipeline.py         orquestación
├── notebooks/
│   └── UrbanRisk_Analytics.ipynb
├── modelos/                artefactos serializados
├── tablero/                tablero y capturas
├── resultados/
│   ├── figuras/
│   └── tablas/
└── docs/
    ├── modelo_de_datos.md  esquema y diccionario de datos
    └── decisiones.md       registro de decisiones metodológicas
```

---

## Modelos implementados

| Tipo | Técnica | Para qué |
|---|---|---|
| Descriptivo | K-Means con selección por codo y silueta | Tipificar alcaldías por perfil de riesgo |
| Descriptivo | DBSCAN con métrica haversine | Detectar concentraciones geográficas sin imponer número de grupos |
| Predictivo | XGBoost, LightGBM, Random Forest, regresión logística multinomial, perceptrón multicapa | Clasificar el nivel de riesgo de cada celda |
| Predictivo | Ensamble por votación suave | Combinar los tres mejores |
| Predictivo | SARIMA con periodo estacional 7 | Pronosticar el volumen diario agregado |
| Prescriptivo | Rejilla de predicción precalculada | Alimentar el tablero de asignación de recursos |

La interpretabilidad se resuelve con SHAP sobre LightGBM. El detalle de por qué no sobre XGBoost está en [`docs/decisiones.md`](docs/decisiones.md).

---

## Decisiones metodológicas

Están razonadas en [`docs/decisiones.md`](docs/decisiones.md). Las cuatro que más condicionan los resultados:

**Reformulación de la unidad de análisis.** El registro del C5 solo contiene eventos: no existe clase negativa. Predecir «accidente / no accidente» era, literalmente, imposible con este dato. Se reconstruyó una rejilla completa de 50 560 filas con 18,57 % de celdas en cero y se discretizó el conteo en terciles de riesgo.

**Validación cronológica.** Ningún corte aleatorio. `TimeSeriesSplit` para la búsqueda de hiperparámetros y un hold-out por fecha en el percentil 80 (2023-09-27) para la evaluación final. Un corte aleatorio sobre una serie temporal deja ver el futuro al modelo.

**Umbrales calculados solo sobre entrenamiento.** Los terciles que definen los niveles de riesgo se estiman en el conjunto de entrenamiento y se aplican al de prueba. Calcularlos sobre el total habría filtrado información.

**Retardos estrictamente hacia atrás.** Las diez variables dinámicas usan exclusivamente información disponible antes del instante predicho.

---

## Limitaciones

- El modelo con retardos **no puede** alimentar la rejilla precalculada del tablero, porque los retardos dependen de la historia reciente y la rejilla se genera con antelación. El motor del tablero es el modelo estructural, marginalmente peor pero precalculable.
- La unidad territorial es la alcaldía. Es gruesa para decisiones operativas: un despliegue real necesitaría resolución de colonia o de malla regular.
- El registro del C5 recoge llamadas de emergencia, no siniestros verificados. Hay sesgo de reporte y subregistro no cuantificado.
- La ventana temporal (2022–2024) incluye la recuperación de movilidad posterior a la pandemia, lo que introduce una tendencia no estacionaria.

---

## Equipo

Equipo 2D — Máster en Ciencia de Datos, UNIR.

## Licencia

MIT. Ver [`LICENSE`](LICENSE).

Los datos de origen pertenecen al Gobierno de la Ciudad de México y se rigen por sus propios términos de uso.
