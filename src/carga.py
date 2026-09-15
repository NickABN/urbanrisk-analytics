# -*- coding: utf-8 -*-
"""Ingesta del registro del C5 y validación de esquema.

Este módulo no transforma nada: lee, comprueba que el archivo tiene la forma
esperada y normaliza el texto. Toda decisión de filtrado vive en
``src.preparacion``, para que la frontera entre «leer» y «decidir» quede clara.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from src import config

#: Columnas sin las cuales el flujo no puede continuar.
COLUMNAS_REQUERIDAS = [
    'fecha_creacion', 'hora_creacion',
    'fecha_cierre', 'hora_cierre',
    'codigo_cierre',
    'alcaldia_catalogo',
    'latitud', 'longitud',
    'tipo_incidente_c4',
    'dia_semana',
]

#: Columnas categóricas que se normalizan a mayúsculas sin acentos.
COLUMNAS_NORMALIZAR = [
    'tipo_incidente_c4', 'incidente_c4',
    'alcaldia_inicio', 'alcaldia_cierre', 'alcaldia_catalogo',
    'colonia_catalogo', 'tipo_entrada', 'clas_con_f_alarma', 'dia_semana',
]


class EsquemaInvalido(ValueError):
    """El archivo de origen no tiene las columnas esperadas."""


def normalizar_texto(s):
    """Mayúsculas sin acentos ni espacios sobrantes.

    Sin esto, «Álvaro Obregón» y «ALVARO OBREGON» se cuentan como dos
    alcaldías distintas al agrupar.
    """
    if pd.isna(s):
        return s
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
    return s.strip().upper()


def validar_esquema(df: pd.DataFrame) -> None:
    """Comprueba que estén las columnas requeridas. Falla pronto y con claridad."""
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise EsquemaInvalido(
            'Faltan columnas obligatorias en el archivo de origen: '
            + ', '.join(faltantes)
            + '. Revisa que el CSV sea el registro del C5 y no otro archivo.'
        )


def cargar(ruta: Path | str | None = None) -> pd.DataFrame:
    """Lee el CSV de origen y devuelve el crudo con marcas temporales y texto normalizado.

    Se lee todo como texto (``dtype=str``) a propósito: la inferencia automática
    de tipos de pandas convierte silenciosamente algunos campos y rompe la
    normalización posterior.
    """
    ruta = Path(ruta) if ruta else config.ruta_csv()
    if not ruta.exists():
        raise FileNotFoundError(
            f'No se encontró el archivo de datos en {ruta}.\n'
            'Define la variable de entorno RUTA_CSV apuntando al CSV completo, '
            'o consulta datos/README.md para obtener la muestra.'
        )

    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_values=['', 'NA'])
    validar_esquema(df)

    df['dt_creacion'] = pd.to_datetime(
        df['fecha_creacion'] + ' ' + df['hora_creacion'], errors='coerce')
    df['dt_cierre'] = pd.to_datetime(
        df['fecha_cierre'] + ' ' + df['hora_cierre'], errors='coerce')

    for c in COLUMNAS_NORMALIZAR:
        if c in df.columns:
            df[c + '_n'] = df[c].map(normalizar_texto)

    return df


def resumen(df: pd.DataFrame) -> dict:
    """Cifras de control de la ingesta, para dejar traza de qué se leyó."""
    return {
        'n_registros': int(len(df)),
        'n_columnas': int(df.shape[1]),
        'dt_creacion_invalidas': int(df['dt_creacion'].isna().sum()),
        'dt_cierre_invalidas': int(df['dt_cierre'].isna().sum()),
        'rango_fechas': [
            str(df['dt_creacion'].min()),
            str(df['dt_creacion'].max()),
        ],
        'codigos_cierre': df['codigo_cierre'].value_counts().to_dict(),
    }


if __name__ == '__main__':
    import json

    datos = cargar()
    print(json.dumps(resumen(datos), ensure_ascii=False, indent=2, default=str))
