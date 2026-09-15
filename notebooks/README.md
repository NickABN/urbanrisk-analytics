# Cuadernos

`UrbanRisk_Analytics.ipynb` — cuaderno maestro. **Pendiente de escribir.** Ver [`../PENDIENTE.md`](../PENDIENTE.md), punto 4.

Es la pieza que responde a la observación del tutor sobre el entregable 3 y la que más pesa en la rúbrica del entregable 4.

## Principio de diseño

El cuaderno **orquesta, no implementa**. Toda la lógica vive en `src/`; el cuaderno importa, ejecuta y narra. Así el código es testeable fuera del cuaderno y el cuaderno se lee como un documento, no como un volcado.

```python
from src import preparacion, caracteristicas, modelos, evaluacion, tablero
R = preparacion.ejecutar()
```

## Reglas

- Primera celda: versiones del entorno vía `src.pipeline.entorno()`.
- Cada sección abre en markdown con **qué se pregunta, cómo se responde y qué significa el resultado**.
- Ninguna celda depende de haber ejecutado otra fuera de orden.
- Por defecto usa la muestra; el CSV completo es opcional con `RUTA_CSV`.
- Se ejecuta de arriba a abajo sin errores antes de guardarse.
