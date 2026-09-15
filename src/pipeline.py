# -*- coding: utf-8 -*-
"""Orquestación del flujo completo.

Uso:

    python -m src.pipeline --todo
    python -m src.pipeline --etapas preparacion caracteristicas
    python -m src.pipeline --todo --sin-shap
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from src import caracteristicas, config, evaluacion, modelos, preparacion, tablero

ETAPAS = {
    'preparacion': preparacion.ejecutar,
    'caracteristicas': caracteristicas.ejecutar,
    'modelos': modelos.ejecutar,
    'evaluacion': evaluacion.ejecutar,
    'tablero': tablero.ejecutar,
}


def entorno() -> dict:
    """Versiones de las librerías críticas.

    Sin esto, un resultado no es reproducible: es una anécdota.
    """
    import importlib

    versiones = {'python': sys.version.split()[0]}
    for nombre in ('pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm',
                   'shap', 'statsmodels', 'matplotlib'):
        try:
            versiones[nombre] = importlib.import_module(nombre).__version__
        except Exception:  # noqa: BLE001
            versiones[nombre] = 'no instalado'
    return versiones


def main() -> int:
    p = argparse.ArgumentParser(description='UrbanRisk Analytics — flujo completo')
    p.add_argument('--todo', action='store_true', help='ejecuta todas las etapas')
    p.add_argument('--etapas', nargs='+', choices=list(ETAPAS),
                   help='ejecuta solo las etapas indicadas, en el orden dado')
    p.add_argument('--sin-shap', action='store_true',
                   help='omite la atribución SHAP, que es la parte más lenta')
    args = p.parse_args()

    if not args.todo and not args.etapas:
        p.print_help()
        return 1

    seleccion = list(ETAPAS) if args.todo else args.etapas

    print('=' * 62)
    print('UrbanRisk Analytics')
    print('=' * 62)
    print('Origen de datos:', config.ruta_csv())
    print('Entorno:', json.dumps(entorno(), ensure_ascii=False))
    print('=' * 62)

    for nombre in seleccion:
        print(f'\n>>> {nombre}')
        t0 = time.time()
        if nombre == 'evaluacion':
            ETAPAS[nombre](con_shap=not args.sin_shap)
        else:
            ETAPAS[nombre]()
        print(f'<<< {nombre} completada en {time.time() - t0:.1f} s')

    print('\nResultados en', config.DIR_RESULTADOS)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
