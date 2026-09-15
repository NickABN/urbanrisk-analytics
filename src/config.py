# -*- coding: utf-8 -*-
"""Configuración central del prototipo.

Todo lo que se repite entre módulos vive aquí: rutas, constantes del dominio
y listas de variables. Ningún otro módulo debe declarar una ruta absoluta.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- rutas ----

RAIZ = Path(__file__).resolve().parent.parent

DIR_DATOS = RAIZ / 'datos'
DIR_MUESTRA = DIR_DATOS / 'muestra'
DIR_INTERMEDIOS = DIR_DATOS / 'intermedios'
DIR_MODELOS = RAIZ / 'modelos'
DIR_RESULTADOS = RAIZ / 'resultados'
DIR_FIGURAS = DIR_RESULTADOS / 'figuras'
DIR_TABLAS = DIR_RESULTADOS / 'tablas'
DIR_TABLERO = RAIZ / 'tablero'

for _d in (DIR_INTERMEDIOS, DIR_MODELOS, DIR_FIGURAS, DIR_TABLAS, DIR_TABLERO):
    _d.mkdir(parents=True, exist_ok=True)


def ruta_csv() -> Path:
    """Devuelve la ruta del CSV de origen.

    Prioriza la variable de entorno ``RUTA_CSV``. Si no está definida,
    recurre a la muestra versionada, que permite ejecutar el flujo completo
    sin descargar los 104 MB del archivo original.
    """
    env = os.environ.get('RUTA_CSV')
    if env:
        return Path(env)
    return DIR_MUESTRA / 'inViales_muestra.csv'


# archivos intermedios
PARQUET_FILTRADO = DIR_INTERMEDIOS / 'datos_filtrado.parquet'
PARQUET_PANEL = DIR_INTERMEDIOS / 'panel_riesgo.parquet'
PARQUET_LAGS = DIR_INTERMEDIOS / 'panel_lags.parquet'

# artefactos
JOBLIB_MODELOS = DIR_MODELOS / 'modelos_riesgo.joblib'
CSV_REJILLA = DIR_TABLERO / 'rejilla_riesgo.csv'

# ------------------------------------------------------------ dominio ----

SEMILLA = 42

ORDEN_DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

FRANJAS = [
    'Madrugada (00-05)',
    'Mañana (06-11)',
    'Tarde (12-17)',
    'Noche (18-23)',
]

#: Solo se conservan los incidentes con este código de cierre (confirmados en sitio).
CODIGO_CIERRE_VALIDO = 'A'

#: Los registros anteriores a este año quedan fuera del periodo declarado.
ANIO_MINIMO = 2022

#: Percentil de la fecha que separa entrenamiento de prueba en la validación
#: cronológica. No se usa partición aleatoria en ningún punto del proyecto.
PERCENTIL_CORTE = 0.8

#: Cuantiles que discretizan el conteo en niveles de riesgo. Se calculan
#: exclusivamente sobre el conjunto de entrenamiento.
CUANTILES_RIESGO = (1 / 3, 2 / 3)

NIVELES_RIESGO = ['Bajo', 'Medio', 'Alto']

# --------------------------------------------------------- variables ----

#: Variables categóricas estructurales.
VAR_CATEGORICAS = ['alcaldia_catalogo', 'franja', 'dia_semana']

#: Variables calendáricas derivadas de la fecha.
VAR_CALENDARIO = ['mes', 'finde', 'trimestre', 'quincena']

#: Variables de retardo. Todas usan exclusivamente información anterior al
#: instante predicho: no hay fuga de información hacia el futuro.
VAR_RETARDO = [
    'lag_1d', 'lag_2d', 'lag_7d', 'lag_14d',
    'mm_7d', 'mm_28d', 'ds_28d',
    'media_hist', 'total_dia_prev', 'lag_franja',
]

#: Conjunto estructural: el que alimenta el tablero, porque es precalculable.
VAR_ESTRUCTURAL = VAR_CATEGORICAS + VAR_CALENDARIO

#: Conjunto completo: estructural más retardos.
VAR_COMPLETO = VAR_CATEGORICAS + VAR_CALENDARIO + VAR_RETARDO

# ------------------------------------------------------------ gráficas ----

ESTILO_FIGURAS = {
    'figure.dpi': 160,
    'font.size': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

COLOR_PRINCIPAL = '#1f4e79'


def franja_de_hora(h: int) -> str:
    """Asigna una hora del día (0–23) a su franja de seis horas."""
    if 0 <= h < 6:
        return FRANJAS[0]
    if 6 <= h < 12:
        return FRANJAS[1]
    if 12 <= h < 18:
        return FRANJAS[2]
    return FRANJAS[3]
