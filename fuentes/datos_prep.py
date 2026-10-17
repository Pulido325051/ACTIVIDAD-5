import re
import pandas as pd
from sklearn.model_selection import train_test_split


def cargar_datos(ruta="datos/datos_ini.csv"):
    # utf-8-sig quita el caracter invisible del inicio del archivo
    return pd.read_csv(ruta, encoding="utf-8-sig")


def limpiar_texto(texto):
    texto = str(texto).lower()
    # Reemplaza los CURP por una palabra generica
    texto = re.sub(r"\b[a-z]{4}\d{6}[hm][a-z]{5}[a-z0-9]\d\b", " curp ", texto)
    # Deja solo letras y espacios
    texto = re.sub(r"[^a-zñ ]", " ", texto)
    # Quita espacios repetidos
    return re.sub(r"\s+", " ", texto).strip()


def limpiar_datos(df, ruta="datos/datos_limp.csv"):
    df = df.dropna(subset=["texto", "categoria"])
    df["texto"] = df["texto"].apply(limpiar_texto)
    # Quita textos vacios y repetidos
    df = df[df["texto"] != ""]
    df = df.drop_duplicates(subset="texto")
    df.to_csv(ruta, index=False)
    return df


def dividir_datos(df):
    # 80% entrenamiento, 20% prueba. La columna "archivo" no se usa
    # porque su nombre revela la clase.
    return train_test_split(df["texto"], df["categoria"], test_size=0.2,
                            random_state=42, stratify=df["categoria"])