# -*- coding: utf-8 -*-
"""UrbanRisk Analytics — prototipo de detección y predicción de riesgo vial.

El flujo canónico es:

    carga -> preparacion -> caracteristicas -> modelos -> evaluacion -> tablero

Cada módulo es ejecutable por separado (``python -m src.<modulo>``) y deja su
salida en disco para que el siguiente la consuma. ``src.pipeline`` los encadena.
"""

__version__ = '1.0.0'
