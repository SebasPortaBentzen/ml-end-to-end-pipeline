# ==========================================
# PROYECTO MÓDULO 6
# Evaluación de un Modelo de Machine Learning
# con Integración de Bases de Datos y Power BI
# ==========================================

import pandas as pd
import os

# ==========================================
# 1. CARGA INICIAL DEL DATASET
# ==========================================

archivo_csv = "dataset_2191_sleep.csv"

# Comprobamos si el archivo existe en la carpeta del proyecto
if not os.path.exists(archivo_csv):
    raise FileNotFoundError(
        f"No se ha encontrado el archivo {archivo_csv}. "
        "Asegúrate de que está en la misma carpeta que este script."
    )

# Cargamos el dataset
df = pd.read_csv(archivo_csv)

print("Dataset cargado correctamente")
print("--------------------------------")
print("Primeras 5 filas:")
print(df.head())

print("\nDimensiones del dataset:")
print(df.shape)

print("\nColumnas del dataset:")
print(df.columns.tolist())

print("\nInformación general:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isnull().sum())

# ==========================================
# 2. LIMPIEZA Y CONVERSIÓN DE DATOS
# ==========================================

import numpy as np

print("\n==========================================")
print("2. LIMPIEZA Y CONVERSIÓN DE DATOS")
print("==========================================")

# Reemplazamos posibles valores problemáticos por NaN
df = df.replace("?", np.nan)

# Columnas que deben ser numéricas
columnas_numericas = [
    "body_weight",
    "brain_weight",
    "max_life_span",
    "gestation_time",
    "predation_index",
    "sleep_exposure_index",
    "danger_index",
    "total_sleep"
]

# Convertimos todas las columnas numéricas
for columna in columnas_numericas:
    df[columna] = pd.to_numeric(df[columna], errors="coerce")

print("\nTipos de datos después de la conversión:")
print(df.dtypes)

print("\nValores nulos después de convertir '?' a NaN:")
print(df.isnull().sum())

# Imputamos valores nulos con la mediana de cada columna
for columna in columnas_numericas:
    mediana = df[columna].median()
    df[columna] = df[columna].fillna(mediana)

print("\nValores nulos después de imputar con la mediana:")
print(df.isnull().sum())

print("\nPrimeras 5 filas después de la limpieza:")
print(df.head())

print("\nLimpieza completada correctamente")

# ==========================================
# 3. GUARDAR DATOS EN BASE DE DATOS SQLITE
# ==========================================

import sqlite3

print("\n==========================================")
print("3. GUARDADO EN SQLITE")
print("==========================================")

# Creamos conexión a la base de datos
conexion = sqlite3.connect("sleep_data.db")

# Guardamos el dataframe como tabla SQL
df.to_sql("sleep_data", conexion, if_exists="replace", index=False)

print("Datos guardados en SQLite correctamente")

# Comprobamos que se ha guardado bien
consulta = "SELECT * FROM sleep_data LIMIT 5"
df_sql = pd.read_sql(consulta, conexion)

print("\nPrimeras filas desde SQLite:")
print(df_sql)

# Cerramos conexión
conexion.close()

print("\nConexión cerrada correctamente")

# ==========================================
# 4. MACHINE LEARNING DESDE SQLITE
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("\n==========================================")
print("4. MACHINE LEARNING DESDE SQLITE")
print("==========================================")

# Volvemos a conectar a la base de datos
conexion = sqlite3.connect("sleep_data.db")

# Leemos datos desde SQL
consulta = """
SELECT 
    body_weight,
    brain_weight,
    max_life_span,
    gestation_time,
    predation_index,
    sleep_exposure_index,
    danger_index,
    total_sleep
FROM sleep_data
"""

df_ml = pd.read_sql(consulta, conexion)

conexion.close()

print("Datos cargados desde SQLite")
print(df_ml.head())

# ==========================================
# VARIABLES
# ==========================================

X = df_ml.drop("total_sleep", axis=1)
y = df_ml["total_sleep"]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTamaño train:", X_train.shape)
print("Tamaño test:", X_test.shape)

# ==========================================
# MODELOS
# ==========================================

modelos = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

resultados = []

print("\nEntrenando modelos...\n")

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    resultados.append({
        "Modelo": nombre,
        "MSE": mse,
        "R2": r2
    })

    print(f"Modelo: {nombre}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}\n")

# Convertimos resultados a DataFrame
df_resultados = pd.DataFrame(resultados)

print("Resultados comparativos:")
print(df_resultados)

# ==========================================
# 5. EXPORTACIÓN PARA POWER BI
# ==========================================

print("\n==========================================")
print("5. EXPORTACIÓN PARA POWER BI")
print("==========================================")

# Usamos el mejor modelo (Decision Tree)
mejor_modelo = DecisionTreeRegressor(random_state=42)
mejor_modelo.fit(X_train, y_train)

y_pred = mejor_modelo.predict(X_test)

# Creamos dataframe comparativo
df_pred = X_test.copy()
df_pred["total_sleep_real"] = y_test.values
df_pred["total_sleep_predicho"] = y_pred

# Añadimos ID único
df_pred = df_pred.reset_index().rename(columns={"index": "id"})

print("\nDatos para Power BI:")
print(df_pred.head())

# Guardamos CSV
df_pred.to_csv("predicciones_powerbi.csv", index=False)

print("\nArchivo 'predicciones_powerbi.csv' creado")

# Guardamos resultados de modelos
df_resultados.to_csv("resultados_modelos.csv", index=False)

print("Archivo 'resultados_modelos.csv' creado")