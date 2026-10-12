import os
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def cargar_datos(ruta="datos/datos_ini.csv"):
    # Si el archivo no existe, descarga el dataset Wine y lo guarda
    if not os.path.exists(ruta):
        os.makedirs("datos", exist_ok=True)
        df = load_wine(as_frame=True).frame
        df.to_csv(ruta, index=False)
    return pd.read_csv(ruta)


def limpiar_datos(df, ruta="datos/datos_limp.csv"):
    # Quita nulos y duplicados y guarda los datos limpios
    df = df.dropna()
    df = df.drop_duplicates()
    df.to_csv(ruta, index=False)
    return df


def dividir_datos(df):
    # 80% entrenamiento, 20% prueba
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)