# -*- coding: utf-8 -*-
"""Rejilla de predicción precalculada para el tablero de inteligencia territorial.

**Por qué existe este módulo.** Un tablero de Power BI no puede invocar un
modelo de Python en tiempo real. La solución es invertir el problema: en vez de
predecir bajo demanda, se enumeran de antemano todas las combinaciones posibles
de la dimensión estructural y se guardan sus predicciones en una tabla que el
tablero consulta por búsqueda.

**Por qué el motor es el modelo estructural y no el mejor modelo.** Las
variables de retardo dependen de la historia reciente, que por definición no se
conoce con antelación. Un modelo con retardos no es precalculable. Se acepta
conscientemente perder catorce milésimas de F1 a cambio de que el tablero
funcione sin infraestructura de inferencia.
"""
from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd

from src import config


def enumerar_combinaciones(alcaldias: list[str]) -> pd.DataFrame:
    """Producto cartesiano de todas las dimensiones estructurales.

    alcaldías × franjas × días de la semana × meses × quincenas.
    El resto de variables calendáricas se deriva de forma determinista.
    """
    filas = itertools.product(
        alcaldias,
        config.FRANJAS,
        config.ORDEN_DIAS,
        range(1, 13),       # mes
        (0, 1),             # quincena
    )
    G = pd.DataFrame(filas, columns=[
        'alcaldia_catalogo', 'franja', 'dia_semana', 'mes', 'quincena'])

    G['finde'] = G['dia_semana'].isin(['Sábado', 'Domingo']).astype(int)
    G['trimestre'] = ((G['mes'] - 1) // 3 + 1).astype(int)
    return G


def ejecutar() -> dict:
    """Genera la rejilla y la deja lista para el tablero."""
    artefactos = joblib.load(config.JOBLIB_MODELOS)
    pipe = artefactos['pipes']['Modelo estructural (XGBoost)']

    panel = pd.read_parquet(config.PARQUET_PANEL)
    alcaldias = sorted(panel['alcaldia_catalogo'].dropna().unique())

    G = enumerar_combinaciones(alcaldias)

    proba = pipe.predict_proba(G[config.VAR_ESTRUCTURAL])
    pred = proba.argmax(axis=1)

    for i, nivel in enumerate(config.NIVELES_RIESGO):
        G[f'prob_{nivel.lower()}'] = proba[:, i].round(4)

    G['nivel_riesgo'] = [config.NIVELES_RIESGO[i] for i in pred]
    G['nivel_codigo'] = pred

    # índice continuo 0–100, más legible en un semáforo que tres probabilidades
    G['indice_riesgo'] = (100 * (proba * np.array([0, .5, 1])).sum(axis=1)).round(1)

    G = G.sort_values(['alcaldia_catalogo', 'mes', 'dia_semana', 'franja'])
    G.to_csv(config.CSV_REJILLA, index=False, encoding='utf-8-sig')

    R = {
        'n_combinaciones': int(len(G)),
        'n_alcaldias': len(alcaldias),
        'tamano_kb': round(config.CSV_REJILLA.stat().st_size / 1024, 1),
        'distribucion_nivel': G['nivel_riesgo'].value_counts().to_dict(),
        'archivo': str(config.CSV_REJILLA),
        'modelo_motor': 'Modelo estructural (XGBoost)',
        'nota': ('Se usa el modelo estructural porque es el único precalculable: '
                 'las variables de retardo dependen de historia reciente.'),
    }

    salida = config.DIR_RESULTADOS / 'tablero.json'
    salida.write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    return R


if __name__ == '__main__':
    print(json.dumps(ejecutar(), ensure_ascii=False, indent=2, default=str))
