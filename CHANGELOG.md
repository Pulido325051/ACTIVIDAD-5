# CHANGELOG

## [1.0.0] - 2026-10-12
### Agregado
- Estructura inicial del repositorio.
- Funciones de carga, limpieza y división de datos (`datos_prep.py`).
- Entrenamiento de regresión logística y random forest con GridSearchCV (5 folds).
- Registro de parámetros, métricas, datos y modelos en MLflow.

### Resultados
| Algoritmo | Mejores hiperparámetros | F1 (CV) | Accuracy | Precision | Recall | F1 (prueba) |
|---|---|---|---|---|---|---|
| Regresión logística | C = 0.1 | 0.993 | 1.00 | 1.00 | 1.00 | 1.00 |
| Random forest | max_depth = 3, n_estimators = 50 | 0.986 | 1.00 | 1.00 | 1.00 | 1.00 |

### Conclusión
Ambos modelos alcanzaron 1.00 en el conjunto de prueba (36 muestras). En validación
cruzada la regresión logística fue ligeramente mejor (0.993 contra 0.986) y es más simple,
por lo que es el modelo elegido. Wine es un dataset pequeño y fácil, así que la diferencia
entre algoritmos es mínima.