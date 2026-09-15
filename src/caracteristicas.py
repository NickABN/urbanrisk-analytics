# -*- coding: utf-8 -*-
"""Ingeniería de variables de retardo temporal.

Las diez variables que construye este módulo son la hipótesis central del
trabajo: que saber qué pasó ayer mejora la predicción de lo que pasará hoy.

El resultado refutó esa hipótesis —la mejora fue del 2,11 %— y el análisis de
ablación en ``src.evaluacion`` explica por qué. El código se conserva porque
el hallazgo negativo *es* el hallazgo.

**Invariante que no se puede romper:** toda variable aquí usa exclusivamente
información anterior al instante que se predice. Un solo ``rolling`` sin
desplazar bastaría para contaminar el modelo con el futuro.
"""
from __future__ import annotations

import json

import pandas as pd

from src import config


def agregar_retardos(X: pd.DataFrame) -> pd.DataFrame:
    """Añade las diez variables dinámicas al panel."""
    X = X.copy()
    X['fecha'] = pd.to_datetime(X['fecha'])
    X['franja'] = pd.Categorical(X['franja'], categories=config.FRANJAS, ordered=True)
    X = X.sort_values(['alcaldia_catalogo', 'franja', 'fecha']).reset_index(drop=True)

    g = X.groupby(['alcaldia_catalogo', 'franja'], observed=True)['n']

    # --- retardos directos de la misma alcaldía y franja ---
    X['lag_1d'] = g.shift(1)
    X['lag_2d'] = g.shift(2)
    X['lag_7d'] = g.shift(7)
    X['lag_14d'] = g.shift(14)

    # --- ventanas móviles sobre el valor ya desplazado ---
    #
    # OJO: `g.shift(1).rolling(...)` NO funciona. El shift devuelve una Series
    # que ha perdido la agrupación, así que la ventana cruzaría de una alcaldía
    # a la siguiente. Hay que reagrupar explícitamente. Este bug estuvo en el
    # código durante una iteración completa.
    X['n_prev'] = g.shift(1)
    gp = X.groupby(['alcaldia_catalogo', 'franja'], observed=True)['n_prev']
    X['mm_7d'] = gp.transform(lambda s: s.rolling(7, min_periods=4).mean())
    X['mm_28d'] = gp.transform(lambda s: s.rolling(28, min_periods=14).mean())
    X['ds_28d'] = gp.transform(lambda s: s.rolling(28, min_periods=14).std())

    # --- media histórica acumulada, solo con pasado ---
    X['media_hist'] = gp.transform(lambda s: s.expanding().mean())

    # --- actividad del día anterior en toda la alcaldía ---
    dia_alc = (X.groupby(['alcaldia_catalogo', 'fecha'], observed=True)['n']
                 .sum().rename('total_dia').reset_index()
                 .sort_values(['alcaldia_catalogo', 'fecha']))
    dia_alc['total_dia_prev'] = dia_alc.groupby('alcaldia_catalogo')['total_dia'].shift(1)
    X = X.merge(dia_alc[['alcaldia_catalogo', 'fecha', 'total_dia_prev']],
                on=['alcaldia_catalogo', 'fecha'], how='left')

    # --- franja inmediatamente anterior; cruza el cambio de día ---
    X = X.sort_values(['alcaldia_catalogo', 'fecha', 'franja']).reset_index(drop=True)
    X['lag_franja'] = X.groupby('alcaldia_catalogo', observed=True)['n'].shift(1)

    return X


def podar_sin_historia(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Descarta las primeras observaciones de cada serie, que carecen de historia.

    Imputarlas sería inventar pasado. Se prefiere perder filas.
    """
    antes = len(X)
    obligatorias = ['lag_14d', 'mm_28d', 'ds_28d',
                    'media_hist', 'lag_franja', 'total_dia_prev']
    X = X.dropna(subset=obligatorias).reset_index(drop=True)
    R = {
        'descartadas_sin_historia': int(antes - len(X)),
        'n_utilizable': int(len(X)),
        'pct_conservado': round(100 * len(X) / antes, 2) if antes else 0.0,
        'rango': [str(X['fecha'].min().date()), str(X['fecha'].max().date())],
    }
    return X, R


def ejecutar() -> dict:
    """Construye el panel con retardos y lo deja en disco."""
    X = pd.read_parquet(config.PARQUET_PANEL)
    R = {'n_panel': int(len(X))}

    X = agregar_retardos(X)
    X, r_poda = podar_sin_historia(X)
    R.update(r_poda)
    R['variables_nuevas'] = config.VAR_RETARDO

    # correlación de cada retardo con el objetivo, para dimensionar de antemano
    # cuánto cabe esperar de ellos
    R['correlacion_con_n'] = {
        c: round(float(X[c].corr(X['n'])), 4) for c in config.VAR_RETARDO
    }

    X.to_parquet(config.PARQUET_LAGS)
    salida = config.DIR_RESULTADOS / 'caracteristicas.json'
    salida.write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    return R


if __name__ == '__main__':
    print(json.dumps(ejecutar(), ensure_ascii=False, indent=2, default=str))
