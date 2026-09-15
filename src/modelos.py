# -*- coding: utf-8 -*-
"""Entrenamiento de los modelos de clasificación con validación cronológica.

Dos reglas gobiernan este módulo y ninguna es negociable:

1. **Nunca una partición aleatoria.** El dato es una serie temporal. Un
   ``train_test_split`` con ``shuffle=True`` deja que el modelo vea el futuro
   y devuelve métricas que no significan nada.
2. **Los umbrales de riesgo se calculan solo sobre entrenamiento.** Discretizar
   el conteo con los terciles del conjunto completo filtraría la distribución
   del periodo de prueba hacia el de entrenamiento.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src import config

warnings.filterwarnings('ignore')


# ------------------------------------------------------------ partición ----

def particion_cronologica(X: pd.DataFrame):
    """Separa entrenamiento y prueba por fecha, no al azar.

    Devuelve ``(entrenamiento, prueba, cortes, resumen)`` donde ``cortes`` son
    los dos umbrales de tercil estimados **solo** sobre entrenamiento.
    """
    X = X.sort_values('fecha').reset_index(drop=True)
    corte = X['fecha'].quantile(config.PERCENTIL_CORTE)

    tr = X[X['fecha'] <= corte].copy()
    te = X[X['fecha'] > corte].copy()

    cortes = tr['n'].quantile(list(config.CUANTILES_RIESGO)).values

    R = {
        'fecha_corte': str(corte.date()),
        'n_entrenamiento': int(len(tr)),
        'n_prueba': int(len(te)),
        'rango_entrenamiento': [str(tr['fecha'].min().date()), str(tr['fecha'].max().date())],
        'rango_prueba': [str(te['fecha'].min().date()), str(te['fecha'].max().date())],
        'umbrales_tercil': [float(c) for c in cortes],
    }
    return tr, te, cortes, R


def etiquetar(serie: pd.Series, cortes) -> np.ndarray:
    """Discretiza el conteo en 0=Bajo, 1=Medio, 2=Alto según los umbrales dados."""
    return np.where(serie <= cortes[0], 0, np.where(serie <= cortes[1], 1, 2))


# ------------------------------------------------------- preprocesamiento ----

def preprocesador(numericas: list[str]) -> ColumnTransformer:
    """Codifica categóricas y estandariza numéricas, imputando por mediana."""
    num = Pipeline([
        ('imputacion', SimpleImputer(strategy='median')),
        ('escalado', StandardScaler()),
    ])
    return ColumnTransformer([
        ('categoricas', OneHotEncoder(handle_unknown='ignore', drop='first'),
         config.VAR_CATEGORICAS),
        ('numericas', num, numericas),
    ])


# ------------------------------------------------------------- catálogo ----

def catalogo(semilla: int = config.SEMILLA) -> list[tuple]:
    """Modelos a comparar: ``(nombre, estimador, variables, numéricas)``.

    El primero es el clasificador trivial. Sin una referencia que batir, un F1
    de 0,67 no dice nada por sí solo.
    """
    CAL = config.VAR_CALENDARIO
    LAG = config.VAR_RETARDO
    EST = config.VAR_ESTRUCTURAL
    COM = config.VAR_COMPLETO

    xgb = dict(subsample=.9, colsample_bytree=.9, random_state=semilla, n_jobs=2,
               tree_method='hist', objective='multi:softprob', num_class=3,
               eval_metric='mlogloss')

    return [
        ('Clasificador trivial (mayoritario)',
         DummyClassifier(strategy='most_frequent'), EST, CAL),

        ('Modelo estructural (XGBoost)',
         XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.1, **xgb), EST, CAL),

        ('Regresión logística con retardos',
         LogisticRegression(max_iter=1500, random_state=semilla), COM, CAL + LAG),

        ('Random Forest con retardos',
         RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_leaf=5,
                                random_state=semilla, n_jobs=2), COM, CAL + LAG),

        ('XGBoost con retardos',
         XGBClassifier(n_estimators=400, max_depth=6, learning_rate=.08, **xgb),
         COM, CAL + LAG),

        ('LightGBM con retardos',
         LGBMClassifier(n_estimators=400, num_leaves=48, learning_rate=.08,
                        subsample=.9, colsample_bytree=.9, random_state=semilla,
                        n_jobs=2, verbose=-1), COM, CAL + LAG),

        ('Red neuronal con retardos',
         MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=350,
                       early_stopping=True, random_state=semilla), COM, CAL + LAG),
    ]


def ensamble(semilla: int = config.SEMILLA) -> VotingClassifier:
    """Votación suave entre los tres modelos de árbol."""
    xgb = dict(subsample=.9, colsample_bytree=.9, random_state=semilla, n_jobs=2,
               tree_method='hist', objective='multi:softprob', num_class=3,
               eval_metric='mlogloss')
    return VotingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=400, max_depth=6, learning_rate=.08, **xgb)),
            ('lgbm', LGBMClassifier(n_estimators=400, num_leaves=48, learning_rate=.08,
                                    subsample=.9, colsample_bytree=.9,
                                    random_state=semilla, n_jobs=2, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=250, max_depth=16,
                                          min_samples_leaf=5, random_state=semilla,
                                          n_jobs=2)),
        ],
        voting='soft')


# ------------------------------------------------------------ evaluación ----

def evaluar(nombre, modelo, variables, numericas, tr, te, ytr, yte, cv=True):
    """Entrena y evalúa un modelo. Devuelve ``(métricas, pipeline, pred, proba)``."""
    pipe = Pipeline([('prep', preprocesador(numericas)), ('mod', modelo)])
    res: dict = {}

    if cv:
        # TimeSeriesSplit, no KFold: los pliegues respetan el orden temporal
        puntajes = cross_val_score(pipe, tr[variables], ytr,
                                   cv=TimeSeriesSplit(n_splits=5),
                                   scoring='f1_macro', n_jobs=1)
        res['cv_f1_media'] = round(float(puntajes.mean()), 4)
        res['cv_f1_desviacion'] = round(float(puntajes.std()), 4)

    pipe.fit(tr[variables], ytr)
    pred = pipe.predict(te[variables])
    proba = pipe.predict_proba(te[variables])

    res.update({
        'exactitud': round(accuracy_score(yte, pred), 4),
        'f1_macro': round(f1_score(yte, pred, average='macro'), 4),
        'precision': round(precision_score(yte, pred, average='macro', zero_division=0), 4),
        'sensibilidad': round(recall_score(yte, pred, average='macro'), 4),
        'auc_ovr': round(roc_auc_score(yte, proba, multi_class='ovr', average='macro'), 4),
        'kappa': round(cohen_kappa_score(yte, pred), 4),
    })
    print(f'{nombre:<38} F1={res["f1_macro"]:.4f}  AUC={res["auc_ovr"]:.4f}', flush=True)
    return res, pipe, pred, proba


def ejecutar() -> dict:
    """Entrena el catálogo completo y serializa los artefactos."""
    X = pd.read_parquet(config.PARQUET_LAGS)
    X['fecha'] = pd.to_datetime(X['fecha'])

    tr, te, cortes, R_part = particion_cronologica(X)
    ytr, yte = etiquetar(tr['n'], cortes), etiquetar(te['n'], cortes)

    R: dict = {'particion': R_part}
    R['balance_entrenamiento'] = (
        pd.Series(ytr).value_counts(normalize=True).round(4).sort_index().to_dict())
    R['balance_prueba'] = (
        pd.Series(yte).value_counts(normalize=True).round(4).sort_index().to_dict())

    resultados, pipes, preds, probas = {}, {}, {}, {}
    for nombre, modelo, variables, numericas in catalogo():
        resultados[nombre], pipes[nombre], preds[nombre], probas[nombre] = evaluar(
            nombre, modelo, variables, numericas, tr, te, ytr, yte)

    nombre = 'Ensamble por votación'
    resultados[nombre], pipes[nombre], preds[nombre], probas[nombre] = evaluar(
        nombre, ensamble(), config.VAR_COMPLETO,
        config.VAR_CALENDARIO + config.VAR_RETARDO, tr, te, ytr, yte, cv=False)

    R['modelos'] = resultados

    candidatos = [k for k in resultados if 'trivial' not in k.lower()]
    mejor = max(candidatos, key=lambda k: resultados[k]['f1_macro'])
    R['mejor'] = mejor
    R['reporte_mejor'] = classification_report(
        yte, preds[mejor], target_names=config.NIVELES_RIESGO,
        output_dict=True, zero_division=0)
    R['matriz_confusion_mejor'] = confusion_matrix(yte, preds[mejor]).tolist()
    R['auc_por_clase'] = {
        c: round(float(roc_auc_score((yte == i).astype(int), probas[mejor][:, i])), 4)
        for i, c in enumerate(config.NIVELES_RIESGO)
    }

    # cuánto aportan realmente los retardos frente al modelo estructural
    base = resultados['Modelo estructural (XGBoost)']['f1_macro']
    top = resultados[mejor]['f1_macro']
    R['aporte_de_los_retardos'] = {
        'f1_estructural': base,
        'f1_mejor': top,
        'diferencia_absoluta': round(top - base, 4),
        'diferencia_relativa_pct': round(100 * (top / base - 1), 2),
    }

    joblib.dump({
        'pipes': pipes, 'preds': preds, 'probas': probas,
        'ytr': ytr, 'yte': yte, 'cortes': cortes,
        'variables_completo': config.VAR_COMPLETO,
        'variables_estructural': config.VAR_ESTRUCTURAL,
        'fecha_corte': R_part['fecha_corte'],
    }, config.JOBLIB_MODELOS)

    salida = config.DIR_RESULTADOS / 'modelos.json'
    salida.write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    return R


if __name__ == '__main__':
    R = ejecutar()
    print('\n=== RESUMEN ===')
    print(json.dumps({k: R[k] for k in
                      ['particion', 'mejor', 'aporte_de_los_retardos', 'auc_por_clase']},
                     ensure_ascii=False, indent=2, default=str))
