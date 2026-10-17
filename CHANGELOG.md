# CHANGELOG

## [2.0.0] - 2026-10-16
### Cambiado
- El dataset pasa de Wine a un clasificador de documentos (4,173 registros, 6 clases, texto por OCR).
- Representación del texto con TF-IDF dentro del Pipeline, en lugar de escalado numérico.
- `datos_prep.py` limpia texto (minúsculas, CURP anonimizado, solo letras, sin duplicados).
- Los CSV dejan de versionarse en Git por contener datos personales; se registra su hash en MLflow.
- Experimento de MLflow: `actividad5_clasificador`.

### Agregado
- ROC-AUC, visualizaciones, matrices de confusión y comparación estadística en el notebook.

### Resultados
| Algoritmo | Mejores hiperparámetros | F1 (CV) | Accuracy | Precision | Recall | F1 (prueba) | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Regresión logística | C = 100 | 0.986 | 0.983 | 0.983 | 0.983 | 0.983 | 0.998 |
| Random forest | max_depth = 20, n_estimators = 200 | 0.987 | 0.989 | 0.989 | 0.989 | 0.989 | 0.999 |

### Conclusión
Ambos modelos superan el 98% de F1. El random forest cometió menos errores en la prueba (9 contra 14 de
834), pero la diferencia no es estadísticamente significativa (Wilcoxon p = 0.28), por lo que se consideran
equivalentes. Los errores se concentran en Certificado Final, Certificado Parcial y Equivalencia, documentos
escolares con vocabulario parecido.

## [1.0.0] - 2026-10-12
### Agregado
- Versión inicial con el dataset Wine, regresión logística y random forest registrados en MLflow.