import hashlib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datos_prep import cargar_datos, limpiar_datos, dividir_datos

# 1. Datos
df = cargar_datos()
df = limpiar_datos(df)
X_train, X_test, y_train, y_test = dividir_datos(df)

# Hash para saber que version de los datos se uso
with open("datos/datos_limp.csv", "rb") as f:
    hash_datos = hashlib.md5(f.read()).hexdigest()

# 2. Algoritmos y sus hiperparametros
modelos = {
    "regresion_logistica": (
        Pipeline([("escalador", StandardScaler()),
                  ("modelo", LogisticRegression(max_iter=1000))]),
        {"modelo__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    ),
}

# 3. Entrenar, ajustar (GridSearchCV con 5 folds) y registrar
mlflow.set_experiment("actividad5_wine")

for nombre, (modelo, parametros) in modelos.items():
    with mlflow.start_run(run_name=nombre):
        busqueda = GridSearchCV(modelo, parametros, cv=5, scoring="f1_weighted")
        busqueda.fit(X_train, y_train)

        mejor = busqueda.best_estimator_
        pred = mejor.predict(X_test)

        mlflow.log_param("algoritmo", nombre)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("hash_datos", hash_datos)
        mlflow.log_params(busqueda.best_params_)

        mlflow.log_metric("cv_f1", busqueda.best_score_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("precision", precision_score(y_test, pred, average="weighted"))
        mlflow.log_metric("recall", recall_score(y_test, pred, average="weighted"))
        mlflow.log_metric("f1", f1_score(y_test, pred, average="weighted"))

        pd.DataFrame(busqueda.cv_results_).to_csv("cv_results.csv", index=False)
        mlflow.log_artifact("cv_results.csv")
        mlflow.log_artifact("datos/datos_limp.csv")
        mlflow.sklearn.log_model(mejor, "modelo", serialization_format="cloudpickle")

        print(nombre, "->", busqueda.best_params_,
              "| cv_f1:", round(busqueda.best_score_, 3),
              "| f1 prueba:", round(f1_score(y_test, pred, average="weighted"), 3))