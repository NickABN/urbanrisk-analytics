# Registro de decisiones metodológicas

Cada entrada explica qué se decidió, por qué, qué se descartó y qué consecuencia tuvo. Las decisiones que salieron mal también están aquí.

---

## D-01 · Reformular la unidad de análisis

**Problema.** El registro del C5 contiene únicamente eventos ocurridos. No hay clase negativa: no existe ninguna fila que diga «aquí no pasó nada». Un clasificador binario «accidente / no accidente» no se puede entrenar con este dato, por mucho que sea lo que el planteamiento inicial pedía.

**Decisión.** Cambiar la unidad de análisis del evento a la **celda espaciotemporal**. Se construye la rejilla completa alcaldía × día × franja mediante un producto cartesiano y se rellenan con cero las combinaciones sin incidentes. El objetivo pasa a ser el **nivel de riesgo** de cada celda, discretizado en terciles del conteo.

**Alternativas descartadas.** Generar pseudonegativos por muestreo aleatorio de tiempo y espacio habría introducido supuestos arbitrarios sobre la exposición. Modelar el conteo directamente con una regresión de Poisson era defendible, pero la sobredispersión la desaconseja —ver D-11 sobre qué cifra corresponde a qué unidad— y el producto operativo pedía niveles, no conteos.

**Consecuencia.** 50 560 observaciones con 18,57 % de ceros. El problema queda bien planteado y la limitación queda documentada con transparencia en la sección 5.1 del TFM.

---

## D-02 · Validación cronológica, nunca aleatoria

**Decisión.** `TimeSeriesSplit` de cinco pliegues para la búsqueda de hiperparámetros y un hold-out por fecha en el percentil 80 (corte en 2023-09-27) para la evaluación final.

**Por qué.** Un `train_test_split` con barajado sobre una serie temporal coloca observaciones del futuro en el conjunto de entrenamiento. Las métricas resultantes son optimistas y no se sostienen en despliegue. Con retardos de hasta catorce días, el problema se agrava: el modelo podría ver literalmente el valor que intenta predecir.

**Consecuencia.** Las métricas son más bajas que con partición aleatoria, y son las correctas.

---

## D-03 · Umbrales de riesgo estimados solo sobre entrenamiento

**Decisión.** Los dos terciles que definen Bajo / Medio / Alto se calculan sobre el conjunto de entrenamiento y se aplican tal cual al de prueba.

**Por qué.** Calcularlos sobre el total filtra la distribución del periodo de prueba hacia el de entrenamiento. Es una fuga sutil, fácil de pasar por alto, y suficiente para inflar las métricas.

**Consecuencia.** El balance de clases en prueba no es exactamente un tercio en cada nivel, y eso es correcto: refleja que el riesgo cambió entre periodos.

---

## D-04 · Retardos estrictamente hacia atrás

**Decisión.** Las diez variables dinámicas usan exclusivamente información anterior al instante predicho.

**Error cometido y corregido.** La primera implementación usaba `g.shift(1).rolling(7).mean()`. El `shift` sobre un objeto agrupado devuelve una Series que ya perdió la agrupación, de modo que la ventana móvil cruzaba de una alcaldía a la siguiente. El código corría sin error y producía resultados plausibles. Se detectó al revisar la continuidad de las series y se corrigió reagrupando explícitamente:

```python
X['n_prev'] = g.shift(1)
gp = X.groupby(['alcaldia_catalogo', 'franja'], observed=True)['n_prev']
X['mm_7d'] = gp.transform(lambda s: s.rolling(7, min_periods=4).mean())
```

**Lección.** Un bug que no lanza excepción y devuelve números razonables es el peor tipo de bug en análisis de datos. La verificación estructural de las series es obligatoria, no opcional.

---

## D-05 · Descartar filas sin historia en vez de imputarlas

**Decisión.** Las observaciones al inicio de cada serie que no tienen los catorce días previos se eliminan.

**Por qué.** Imputar un retardo con la mediana equivale a afirmar que el pasado fue el promedio, que es precisamente lo que el modelo intenta aprender. La imputación introduciría una señal circular.

**Consecuencia.** Se pierden filas al comienzo del periodo. El porcentaje conservado se reporta en `resultados/caracteristicas.json`.

---

## D-06 · SHAP sobre LightGBM y no sobre XGBoost

**Problema.** `shap.TreeExplainer` falla sobre el XGBoost multiclase con `ValueError: could not convert string to float: '[3.64e-2,...]'`. La causa es que XGBoost genera un `base_score` vectorial en problemas multiclase y la versión 0.46 de `shap` espera un escalar. Fijar `base_score=0.5` manualmente no resolvió el problema.

**Decisión.** Sustituir por LightGBM con configuración equivalente para la parte de interpretabilidad.

**Justificación.** El rendimiento es prácticamente idéntico (F1 0,6656 frente a 0,6666), así que las atribuciones son representativas del modelo evaluado. La alternativa —usar `KernelExplainer`, agnóstico al modelo— era órdenes de magnitud más lenta y aproximada.

**Consecuencia.** Se documenta explícitamente en el TFM que la atribución se calcula sobre un modelo sustituto. No se oculta.

---

## D-07 · El motor del tablero es el modelo estructural, aun siendo peor

**Problema.** El mejor modelo incorpora retardos. Los retardos dependen de la historia reciente. La historia reciente no se conoce cuando se precalcula una rejilla.

**Decisión.** El tablero se alimenta del **modelo estructural**, que usa solo alcaldía, franja, día de la semana y variables calendáricas. Todas conocidas con arbitraria antelación.

**Coste.** Catorce milésimas de F1 (0,657 frente a 0,671).

**Justificación.** A cambio se obtiene un tablero que funciona sin servidor de inferencia, sin latencia y sin dependencias de Python en el lado del cliente. Para el uso previsto —planificación de despliegue de recursos con días de anticipación— la pérdida es irrelevante. Si el caso de uso cambiara a alerta en tiempo real, la decisión debería revisarse.

---

## D-08 · Reportar que la hipótesis falló

**Hipótesis inicial.** Incorporar variables de retardo temporal mejoraría sustancialmente la predicción del riesgo.

**Resultado.** Mejoró un 2,11 % relativo. Catorce milésimas de F1.

**Decisión.** Reportarlo como hallazgo principal en vez de enterrarlo, y dedicar esfuerzo a explicar **por qué**.

**Evidencia reunida.**

| Configuración de variables | F1 macro |
|---|---|
| Completo (zona + calendario + retardos) | 0,6666 |
| Solo estructura (zona + calendario) | 0,6576 |
| Solo dinámica (retardos) | 0,6565 |
| Zona + retardos (sin calendario) | 0,6592 |
| Calendario + retardos (sin zona) | 0,6582 |

Cada mitad alcanza casi lo mismo que el conjunto. Las dos fuentes de información son redundantes: saber que es viernes por la noche en Iztapalapa ya contiene casi todo lo que aporta saber qué pasó el viernes pasado. A esto se suman la sobredispersión (D-11) y asociaciones débiles en todas las categóricas (V de Cramér < 0,11).

**Techo de lo predecible.** Un oráculo que conoce la media histórica exacta de cada celda —información que ningún modelo real tiene— alcanza F1 = 0,5921 y exactitud 0,5957. Los modelos entrenados la superan, lo que indica que están extrayendo señal más allá del promedio; pero el margen total disponible es estrecho.

**Nota de implementación.** Estas cinco cifras salen de **XGBoost** con `n_estimators=400, max_depth=6, learning_rate=0.08`. El estimador no es intercambiable: sustituirlo por LightGBM con configuración análoga produce valores sistemáticamente más bajos (≈0,654 / 0,656 / 0,658) y rompe la correspondencia con el documento. Los bloques puramente numéricos —solo retardos, y calendario más retardos— usan un pipeline sin `OneHotEncoder`, porque no hay nada que codificar.

**Conclusión.** El techo de F1 ≈ 0,67 es **estructural, no algorítmico**. Ningún modelo adicional lo superaría de forma apreciable con este dato y esta granularidad. Superarlo requiere dato nuevo —aforos vehiculares, clima, eventos masivos, obra vial— o granularidad más fina.

---

## D-09 · Filtrar por código de cierre `A`

**Decisión.** Conservar solo los incidentes confirmados en sitio.

**Por qué.** Los demás códigos incluyen falsas alarmas, duplicados y llamadas informativas. Contarlos como incidentes inflaría el conteo con ruido que no corresponde a siniestros reales.

**Coste.** Se reduce el volumen de trabajo de forma sustancial. Se acepta a cambio de que cada fila signifique lo que dice significar.

---

## D-10 · Franjas de seis horas

**Decisión.** Cuatro bloques: madrugada, mañana, tarde y noche.

**Por qué.** Coinciden con los turnos operativos de los cuerpos de emergencia, de modo que una predicción por franja se traduce directamente en una decisión de asignación. Una granularidad horaria habría multiplicado por seis el número de celdas y elevado la proporción de ceros hasta hacer el problema degenerado.

**Consecuencia.** Se pierde resolución intradía. La franja de tarde (12–17 h) concentra buena parte de la actividad y su interior no se distingue.

---

## D-11 · La sobredispersión: dos cifras, dos unidades

**Qué se detectó.** El documento cita una razón varianza/media de **250,78** como argumento para descartar la regresión de Poisson. Al reproducir el flujo, la rejilla de modelado arroja **3,44**. La diferencia es de dos órdenes de magnitud.

**De dónde sale cada una.** Ambas están bien calculadas; miden cosas distintas.

| Unidad de agregación | Celdas | Media | Razón var/media |
|---|---|---|---|
| Rejilla de modelado: alcaldía × **fecha** × franja | 50 560 | ≈ 3,6 | **3,44** |
| Agregado: alcaldía × franja × **día de la semana** | 448 | 410,44 | **250,78** |

El 250,78 procede de `07_eda_supuestos.py`, que agrupa por día de la semana como categoría —lunes, martes…— y por tanto colapsa las ~113 ocurrencias de cada día en una sola celda.

**Por qué importa la distinción.** La razón calculada sobre el agregado está dominada por la **heterogeneidad entre alcaldías**: Iztapalapa y Milpa Alta tienen medias muy distintas, y esa variación entre celdas se cuela en el numerador. No es sobredispersión de un proceso de conteo; es varianza entre poblaciones distintas. Al agregar sobre 113 fechas, la señal territorial se amplifica y la dispersión intrínseca se diluye.

La cifra pertinente para decidir si Poisson aplica **al modelo que efectivamente se entrena** es la de la rejilla: 3,44.

**Estado.** El texto del apartado 4.7 dice «los conteos agregados», así que no afirma nada falso sobre lo que midió. Lo que hace es usar una magnitud medida sobre una unidad para justificar una decisión de modelado tomada sobre otra. La conclusión sobrevive —3,44 también rechaza la equidispersión, y con holgura—, pero la magnitud está sobredimensionada por un factor de setenta.

**Recomendación.** Corregirlo antes de la entrega final. Citar 3,44 sobre la rejilla como argumento contra Poisson, y conservar el 250,78 como observación separada sobre la heterogeneidad territorial, que es lo que realmente demuestra. `src.preparacion.sobredispersion()` calcula las dos y las etiqueta.

**Lección.** Una cifra correcta puede sostener un argumento equivocado si se calcula sobre una unidad y se aplica a otra. Fue el propio flujo reproducible el que lo sacó a la luz: es exactamente para esto que sirve un prototipo que recrea los resultados.
