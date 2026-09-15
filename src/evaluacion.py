# -*- coding: utf-8 -*-
"""Evaluación, ablación, calibración e interpretabilidad.

Este módulo produce la evidencia del hallazgo central del trabajo: que el techo
predictivo es estructural y no algorítmico. La pieza decisiva es la ablación,
que compara tres configuraciones de variables sobre la misma partición.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src import config, modelos

warnings.filterwarnings('ignore')


# ------------------------------------------------------------- ablación ----

#: Hiperparámetros del estimador de la ablación.
#:
#: Es XGBoost, no LightGBM, y eso importa: las cifras publicadas en el TFM
#: (0,6576 / 0,6565 / 0,6666) salieron de este estimador exacto. Sustituirlo
#: por LightGBM produce valores sistemáticamente más bajos y rompe la
#: correspondencia con el documento.
XGB_ABLACION = dict(
    n_estimators=400, max_depth=6, learning_rate=.08,
    subsample=.9, colsample_bytree=.9, n_jobs=2,
    tree_method='hist', objective='multi:softprob',
    num_class=3, eval_metric='mlogloss',
)


def _pipeline_ablacion(variables, numericas, semilla):
    """Construye el pipeline según haya o no variables categóricas.

    Cuando el bloque es puramente numérico —solo retardos, o calendario más
    retardos— no hay nada que codificar, así que se usa un pipeline simple en
    lugar de un ``ColumnTransformer``. Meter un OneHotEncoder vacío cambiaría
    el resultado.
    """
    hay_categoricas = any(v in config.VAR_CATEGORICAS for v in variables)
    if hay_categoricas:
        prep = modelos.preprocesador(numericas)
    else:
        prep = Pipeline([
            ('imputacion', SimpleImputer(strategy='median')),
            ('escalado', StandardScaler()),
        ])
    return Pipeline([
        ('prep', prep),
        ('mod', XGBClassifier(random_state=semilla, **XGB_ABLACION)),
    ])


def ablacion(tr, te, ytr, yte, semilla: int = config.SEMILLA) -> dict:
    """Descompone el aporte de cada bloque de información.

    Cinco configuraciones sobre la misma partición. Si el conjunto completo
    apenas supera a cada mitad por separado, las dos fuentes son redundantes
    entre sí — que es exactamente lo que ocurre, y la razón por la que los
    retardos no aportan.

    Los cinco bloques y el oráculo reproducen el análisis original del
    entregable 3 (``26b_ablacion.py``). No reducir el número de bloques: los
    dos intermedios son los que permiten separar el aporte de la zona del
    aporte del calendario.
    """
    CAT, CAL, LAG = config.VAR_CATEGORICAS, config.VAR_CALENDARIO, config.VAR_RETARDO

    bloques = {
        'Completo (zona + calendario + retardos)': (config.VAR_COMPLETO, CAL + LAG),
        'Solo estructura (zona + calendario)': (config.VAR_ESTRUCTURAL, CAL),
        'Solo dinámica (retardos)': (LAG, LAG),
        'Zona + retardos (sin calendario)': (CAT + LAG, LAG),
        'Calendario + retardos (sin zona)': (CAL + LAG, CAL + LAG),
    }

    resultados = {}
    for nombre, (variables, numericas) in bloques.items():
        pipe = _pipeline_ablacion(variables, numericas, semilla)
        pipe.fit(tr[variables], ytr)
        pred = pipe.predict(te[variables])
        resultados[nombre] = round(f1_score(yte, pred, average='macro'), 4)
        print(f'{nombre:<42} F1={resultados[nombre]:.4f}', flush=True)

    completo = resultados['Completo (zona + calendario + retardos)']
    resultados['ganancia_sobre_estructura'] = round(
        completo - resultados['Solo estructura (zona + calendario)'], 4)
    resultados['ganancia_sobre_dinamica'] = round(
        completo - resultados['Solo dinámica (retardos)'], 4)
    return resultados


def oraculo_media_historica(te, yte, cortes) -> dict:
    """Techo de lo predecible: un predictor que conoce la media histórica exacta.

    Responde a «¿cuánta de la variación es siquiera predecible?». Si un oráculo
    que ya sabe el promedio histórico de cada celda no supera cierto F1, ningún
    modelo que estime ese promedio puede superarlo tampoco.
    """
    mh = te['media_hist'].values
    pred = np.where(mh <= cortes[0], 0, np.where(mh <= cortes[1], 1, 2))
    from sklearn.metrics import accuracy_score
    return {
        'f1_macro': round(float(f1_score(yte, pred, average='macro')), 4),
        'exactitud': round(float(accuracy_score(yte, pred)), 4),
    }


# ---------------------------------------------------------- calibración ----

def calibracion(yte, proba, n_bins: int = 10) -> dict:
    """Mide si las probabilidades predichas son creíbles como probabilidades.

    Un modelo puede ordenar bien y aun así estar mal calibrado. Para un tablero
    que muestra «probabilidad de riesgo alto», la calibración importa tanto
    como la exactitud.
    """
    R = {'log_loss': round(float(log_loss(yte, proba)), 4), 'por_clase': {}}

    for i, nivel in enumerate(config.NIVELES_RIESGO):
        binario = (np.asarray(yte) == i).astype(int)
        frac, media = calibration_curve(binario, proba[:, i], n_bins=n_bins,
                                        strategy='quantile')
        R['por_clase'][nivel] = {
            'brier': round(float(brier_score_loss(binario, proba[:, i])), 4),
            'curva_observado': [round(float(v), 4) for v in frac],
            'curva_predicho': [round(float(v), 4) for v in media],
        }
    return R


# ------------------------------------------------------ interpretabilidad ----

def atribucion_shap(tr, te, ytr, muestra: int = 4000,
                    semilla: int = config.SEMILLA) -> dict:
    """Importancia de variables por valores de Shapley.

    Se usa LightGBM y no XGBoost a propósito. ``shap`` 0.46 no admite el vector
    ``base_score`` que XGBoost genera en problemas multiclase y falla con un
    error de conversión. LightGBM da un rendimiento equivalente (F1 0,6656
    frente a 0,6666), así que el cambio no altera las conclusiones.
    """
    import shap

    variables = config.VAR_COMPLETO
    numericas = config.VAR_CALENDARIO + config.VAR_RETARDO

    pipe = Pipeline([
        ('prep', modelos.preprocesador(numericas)),
        ('mod', LGBMClassifier(n_estimators=400, num_leaves=48, learning_rate=.08,
                               subsample=.9, colsample_bytree=.9,
                               random_state=semilla, n_jobs=2, verbose=-1)),
    ])
    pipe.fit(tr[variables], ytr)

    prep = pipe.named_steps['prep']
    nombres = list(prep.get_feature_names_out())
    sub = te[variables].sample(min(muestra, len(te)), random_state=semilla)
    matriz = prep.transform(sub)

    explicador = shap.TreeExplainer(pipe.named_steps['mod'])
    valores = explicador.shap_values(matriz)

    if isinstance(valores, list):
        importancia = np.mean([np.abs(v).mean(axis=0) for v in valores], axis=0)
    else:
        importancia = np.abs(valores).mean(axis=(0, 2)) if valores.ndim == 3 \
            else np.abs(valores).mean(axis=0)

    orden = np.argsort(importancia)[::-1]
    return {
        'n_muestra': int(len(sub)),
        'importancia': {nombres[i]: round(float(importancia[i]), 6)
                        for i in orden[:30]},
    }


# -------------------------------------------------------------- flujo ----

def ejecutar(con_shap: bool = True) -> dict:
    """Evaluación completa sobre los artefactos ya entrenados."""
    X = pd.read_parquet(config.PARQUET_LAGS)
    X['fecha'] = pd.to_datetime(X['fecha'])

    tr, te, cortes, R_part = modelos.particion_cronologica(X)
    ytr = modelos.etiquetar(tr['n'], cortes)
    yte = modelos.etiquetar(te['n'], cortes)

    R: dict = {'particion': R_part}

    print('--- ablación ---')
    R['ablacion'] = ablacion(tr, te, ytr, yte)

    print('--- oráculo de la media histórica ---')
    R['oraculo_media_historica'] = oraculo_media_historica(te, yte, cortes)
    print(R['oraculo_media_historica'], flush=True)

    artefactos = joblib.load(config.JOBLIB_MODELOS)
    resumen = json.loads((config.DIR_RESULTADOS / 'modelos.json').read_text(encoding='utf8'))
    mejor = resumen['mejor']
    R['modelo_evaluado'] = mejor

    print('--- calibración ---')
    R['calibracion'] = calibracion(yte, artefactos['probas'][mejor])

    if con_shap:
        print('--- atribución SHAP ---')
        try:
            R['shap'] = atribucion_shap(tr, te, ytr)
        except Exception as e:  # noqa: BLE001
            R['shap'] = {'error': str(e)}
            print(f'SHAP no disponible: {e}')

    salida = config.DIR_RESULTADOS / 'evaluacion.json'
    salida.write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    return R


if __name__ == '__main__':
    R = ejecutar()
    print('\n=== ABLACIÓN ===')
    print(json.dumps(R['ablacion'], ensure_ascii=False, indent=2))
