# -*- coding: utf-8 -*-
"""Limpieza y construcción de la unidad de análisis.

Aquí ocurre la decisión que define el proyecto entero: el registro del C5
contiene **solo eventos**, de modo que no existe clase negativa y predecir
«accidente / no accidente» es imposible con este dato.

La salida es una rejilla completa **alcaldía × día × franja horaria** en la que
las combinaciones sin ningún incidente aparecen explícitamente con conteo cero.
Eso crea la variación necesaria para un problema de aprendizaje supervisado
bien planteado.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import carga, config


def filtrar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica los dos filtros del proyecto y devuelve el subconjunto de trabajo.

    1. Se descartan los registros anteriores a 2022, fuera del periodo declarado.
    2. Se conservan solo los incidentes con código de cierre ``A``, es decir,
       confirmados en sitio. El resto son falsas alarmas, duplicados y llamadas
       informativas que inflarían el conteo con ruido.
    """
    R: dict = {'n_crudo': int(len(df))}

    df = df.copy()
    df['anio'] = df['dt_creacion'].dt.year
    R['n_fuera_de_periodo'] = int((df['anio'] < config.ANIO_MINIMO).sum())
    df = df[df['anio'] >= config.ANIO_MINIMO]
    R['n_en_periodo'] = int(len(df))

    d = df[df['codigo_cierre'] == config.CODIGO_CIERRE_VALIDO].copy()
    R['n_confirmados'] = int(len(d))
    R['pct_confirmados'] = round(100 * len(d) / len(df), 2) if len(df) else 0.0

    return d, R


def derivar(d: pd.DataFrame) -> pd.DataFrame:
    """Añade las variables temporales y geográficas que usa el resto del flujo."""
    d = d.copy()
    d['hora'] = d['dt_creacion'].dt.hour
    d['franja'] = d['hora'].map(config.franja_de_hora)
    d['fecha'] = d['dt_creacion'].dt.date
    d['mes'] = d['dt_creacion'].dt.month
    d['semana'] = d['dt_creacion'].dt.isocalendar().week.astype(int)
    d['anio'] = d['dt_creacion'].dt.year
    d['dia_semana'] = pd.Categorical(
        d['dia_semana'], categories=config.ORDEN_DIAS, ordered=True)
    d['tiempo_atencion_min'] = (
        d['dt_cierre'] - d['dt_creacion']).dt.total_seconds() / 60
    d['lon'] = pd.to_numeric(d['longitud'], errors='coerce')
    d['lat'] = pd.to_numeric(d['latitud'], errors='coerce')
    return d


def construir_rejilla(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construye la rejilla completa alcaldía × día × franja.

    El punto clave es el ``reindex`` contra el producto cartesiano: sin él,
    las celdas sin incidentes simplemente no existirían y el conteo estaría
    condicionado a que hubiera ocurrido algo.
    """
    conteos = (d.dropna(subset=['alcaldia_catalogo'])
                 .groupby(['alcaldia_catalogo', 'fecha', 'franja'])
                 .size().reset_index(name='n'))

    alcaldias = sorted(d['alcaldia_catalogo'].dropna().unique())
    fechas = pd.date_range(
        d['dt_creacion'].min().date(), d['dt_creacion'].max().date(), freq='D').date

    idx = pd.MultiIndex.from_product(
        [alcaldias, fechas, config.FRANJAS],
        names=['alcaldia_catalogo', 'fecha', 'franja'])

    X = (conteos.set_index(['alcaldia_catalogo', 'fecha', 'franja'])
                .reindex(idx, fill_value=0)
                .reset_index())

    X['fecha'] = pd.to_datetime(X['fecha'])
    X['dia_semana'] = X['fecha'].dt.dayofweek.map(dict(enumerate(config.ORDEN_DIAS)))
    X['mes'] = X['fecha'].dt.month
    X['anio'] = X['fecha'].dt.year
    X['finde'] = (X['fecha'].dt.dayofweek >= 5).astype(int)
    X['trimestre'] = X['fecha'].dt.quarter
    X['dia_del_anio'] = X['fecha'].dt.dayofyear
    X['quincena'] = (X['fecha'].dt.day >= 15).astype(int)

    R = {
        'n_alcaldias': len(alcaldias),
        'n_dias': len(fechas),
        'n_franjas': len(config.FRANJAS),
        'n_observaciones': int(len(X)),
        'n_celdas_en_cero': int((X['n'] == 0).sum()),
        'pct_celdas_en_cero': round(100 * (X['n'] == 0).mean(), 2),
    }
    return X, R


def sobredispersion(d: pd.DataFrame, X: pd.DataFrame) -> dict:
    """Razón varianza/media sobre las dos unidades de agregación del proyecto.

    Se calculan las dos a propósito, porque el TFM cita una y modela sobre la
    otra, y la diferencia es de dos órdenes de magnitud.

    - **Rejilla de modelado** (alcaldía × fecha × franja): es la unidad sobre la
      que se entrena. Es la cifra pertinente para decidir si una regresión de
      Poisson es aplicable *a este modelo*.
    - **Agregado por día de la semana** (alcaldía × franja × día de la semana):
      colapsa las ~113 ocurrencias de cada día de la semana en una sola celda.
      La razón que produce está dominada por la heterogeneidad entre alcaldías
      —Iztapalapa frente a Milpa Alta—, no por la dispersión de un proceso de
      conteo. Es la cifra que aparece en el apartado 4.7 del documento.

    Ambas superan la unidad, así que la conclusión —Poisson simple no
    aplica— se sostiene en los dos casos. Lo que no se sostiene es citar la
    magnitud de una como si describiera la otra.
    """
    def razon(serie: pd.Series) -> dict:
        return {
            'media': round(float(serie.mean()), 2),
            'varianza': round(float(serie.var(ddof=1)), 2),
            'razon_var_media': round(float(serie.var(ddof=1) / serie.mean()), 2),
        }

    agregado = (d.groupby(['alcaldia_catalogo', 'franja', 'dia_semana'], observed=True)
                  .size())

    return {
        'rejilla_modelado': razon(X['n']),
        'agregado_por_dia_semana': razon(agregado),
        'nota': ('La razón sobre el agregado está inflada por la heterogeneidad '
                 'entre alcaldías, no por sobredispersión del proceso de conteo. '
                 'Para argumentar sobre el modelo, usar la de la rejilla.'),
    }


def ejecutar() -> dict:
    """Flujo completo de preparación. Deja dos parquet y devuelve las cifras."""
    crudo = carga.cargar()
    R = {'ingesta': carga.resumen(crudo)}

    d, r_filtro = filtrar(crudo)
    R['filtrado'] = r_filtro

    d = derivar(d)
    d.to_parquet(config.PARQUET_FILTRADO)

    X, r_rejilla = construir_rejilla(d)
    R['rejilla'] = r_rejilla
    R['sobredispersion'] = sobredispersion(d, X)
    X.to_parquet(config.PARQUET_PANEL)

    salida = config.DIR_RESULTADOS / 'preparacion.json'
    salida.write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    return R


if __name__ == '__main__':
    print(json.dumps(ejecutar(), ensure_ascii=False, indent=2, default=str))
