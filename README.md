# Actividad 5: Clasificador de documentos con MLflow

## Objetivo
Preparar un dataset propio, entrenar dos algoritmos de clasificación, ajustar hiperparámetros
con validación cruzada y registrar los experimentos en MLflow.

## Dataset
Documentos escolares y de identidad digitalizados (PDF) cuyo texto se obtuvo por OCR:
4,173 registros con las columnas `texto`, `categoria` y `archivo`, repartidos en 6 clases casi
balanceadas (Actas de nacimiento, CURP, Certificado Final, Certificado Parcial, Certificados
Secundaria y Equivalencia).

**Los datos no se incluyen en este repositorio.** El texto contiene datos personales de personas
reales, por lo que `datos/datos_ini.csv` y `datos/datos_limp.csv` se mantienen fuera de Git
(están en `.gitignore`). La versión de los datos usada en cada corrida queda identificada con el
hash MD5 del archivo limpio, registrado en MLflow como `hash_datos`.

## Estructura
- `datos/`: carpeta donde se coloca `datos_ini.csv` (datos originales, no incluido) y donde se
  genera `datos_limp.csv` (datos limpios, no incluido)
- `fuentes/datos_prep.py`: funciones de carga, limpieza y división
- `fuentes/train.py`: entrenamiento y registro en MLflow
- `fuentes/entrena.ipynb`: notebook de Colab con entrenamiento, evaluación y gráficas
- `evidencias/`: capturas de la interfaz de MLflow
- `CHANGELOG.md`: reporte final con resultados y conclusión

## Cómo reproducirlo
1. Clonar el repositorio e instalar dependencias:
```bash
git clone https://github.com/Pulido325051/ACTIVIDAD-5.git
cd ACTIVIDAD-5
pip install mlflow scikit-learn pandas
```
2. Colocar el dataset (privado) en `datos/datos_ini.csv`, con las columnas `texto`, `categoria`
   y `archivo`.
3. Ejecutar el entrenamiento y abrir la interfaz de MLflow:
```bash
python fuentes/train.py
python -m mlflow ui
```
Abrir http://127.0.0.1:5000 para ver el experimento `actividad5_clasificador`.

En Google Colab, abrir `fuentes/entrena.ipynb` y ejecutar las celdas en orden; la última celda del
paso 1 pide subir `datos_ini.csv`.

## Decisiones
- Algoritmos: regresión logística y random forest.
- Representación del texto: TF-IDF (5,000 términos) dentro de un `Pipeline`, de modo que se ajusta
  solo con los datos de entrenamiento en cada fold.
- Limpieza: minúsculas, reemplazo de los CURP por una palabra genérica, solo letras y espacios, y
  eliminación de textos vacíos y repetidos (6), quedando 4,167 registros.
- La columna `archivo` no se usa como variable: su nombre revela la clase y causaría fuga de información.
- Validación: GridSearchCV con 5 folds y F1 ponderado.
- División 80/20 estratificada, con `random_state = 42`.
- Métricas: accuracy, precision, recall, F1 y ROC-AUC (uno contra el resto).

## Resultados
Ejecución del notebook en Google Colab, sobre el conjunto de prueba (834 documentos):

| Algoritmo | Mejores hiperparámetros | F1 (CV) | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión logística | C = 100 | 0.986 | 0.983 | 0.983 | 0.998 |
| Random forest | max_depth = 20, n_estimators = 200 | 0.987 | 0.989 | 0.989 | 0.999 |

La comparación estadística (validación cruzada repetida 5x3, t pareada p = 0.30 y Wilcoxon
p = 0.28) no muestra diferencia significativa entre ambos modelos. El detalle y la conclusión
están en `CHANGELOG.md`.

Nota de reproducibilidad: las capturas de MLflow provienen de una ejecución local con `train.py`,
y los valores pueden variar en milésimas respecto a Colab por las versiones de las librerías.

## Evidencias de MLflow
Se configuró MLflow con seguimiento local y se validó que cada corrida registre parámetros,
métricas y artefactos (`cv_results.csv`, `datos_limp.csv` y el modelo).

**Experimento `actividad5_clasificador`**

![Inicio de MLflow](evidencias/01_inicio.png)

**Corridas del experimento (una por algoritmo)**

![Lista de corridas](evidencias/02_corridas.png)

**Métricas y parámetros registrados**

![Métricas y parámetros](evidencias/03_metricas_parametros.png)

**Detalle de una corrida**

![Detalle de la corrida](evidencias/04_detalle_corrida.png)

**Artefactos registrados**

![Artefactos](evidencias/05_artefactos.png)