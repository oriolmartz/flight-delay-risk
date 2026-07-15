# Guía de métricas de la interfaz

Flight Delay Risk está diseñado para entenderse sin leer la documentación del repositorio.

## Métricas visibles de decisión

| Etiqueta | Significado | Dirección |
|---|---|---|
| Riesgo estimado de retraso | Probabilidad estimada de llegar con al menos 15 minutos de retraso. | Depende del contexto |
| Riesgo comparado con esta ruta | Estimación actual dividida por la tasa histórica de retraso de la ruta. | Más de 1× significa riesgo superior al habitual |
| Evidencia detrás de la referencia | Número de vuelos anteriores usados para construir la comparación. | Más evidencia suele ser mejor |
| Calidad del ranking | Traducción de PR-AUC: capacidad de situar los vuelos retrasados arriba. | Cuanto mayor, mejor |
| Ventaja de la lista prioritaria | Lift del top 10% frente a revisar vuelos al azar. | Cuanto mayor, mejor |
| Diferencia de fiabilidad | Desajuste medio entre riesgo predicho y resultados observados. | Cuanto menor, mejor |

## Revelado progresivo

La vista principal responde cuatro preguntas:

1. ¿Cuál es el riesgo estimado?
2. ¿Es inusual para esta ruta?
3. ¿Cuánta evidencia histórica sostiene la comparación?
4. ¿Qué factores movieron la estimación?

Scores brutos, calibradores, Brier, ECE, diagnósticos completos por fold y métricas de runtime quedan en desplegables **Avanzados**.

## Límite de interpretación

Las contribuciones describen asociaciones aprendidas por el modelo. No demuestran causas reales del retraso.
