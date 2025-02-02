import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📌 Función para cargar los datos
def cargar_datos(ruta="dataset/ames_housing.csv"):
    try:
        df = pd.read_csv(ruta)
        print("✅ Datos cargados correctamente.")
        return df
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None

# 📌 Función para preprocesar los datos (limpieza y división en train/test)
def preprocesar_datos(df):
    df.columns = df.columns.str.replace(" ", "")  # Elimina espacios en nombres de columnas
    
    if "GrLivArea" not in df.columns or "SalePrice" not in df.columns:
        raise KeyError("⚠️ Error: Las columnas necesarias no están en el dataset.")
    
    X = df[["GrLivArea"]]  # Variable independiente
    y = df["SalePrice"]     # Variable dependiente
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Datos preprocesados correctamente.")
    return X_train, X_test, y_train, y_test

# 📌 Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print("✅ Modelo entrenado correctamente.")
    return modelo

# 📌 Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Evaluación del modelo:")
    print(f"📉 Error cuadrático medio (MSE): {mse:.2f}")
    print(f"📈 Coeficiente de determinación (R²): {r2:.4f}")
    
    return mse, r2

# 📌 Función para realizar una predicción con un dato aleatorio de prueba
def prediccion_aleatoria(modelo, X_test, y_test):
    muestra = X_test.sample(1)  # Seleccionar un registro al azar de los datos de prueba
    area = muestra  # Ya es un DataFrame, no es necesario reshaping
    precio_real = y_test.loc[muestra.index].values[0]
    
    precio_predicho = modelo.predict(area)[0]
    
    print("\n🎯 Predicción para una casa:")
    print(f"🏡 Área: {area.iloc[0, 0]:,.2f} sqft")  # Usamos iloc para acceder por posición
    print(f"🔹 Precio real: ${precio_real:,.2f}")
    print(f"🔹 Precio predicho: ${precio_predicho:,.2f}")

# 📌 Función para guardar el modelo entrenado
def guardar_modelo(modelo, ruta="modelo.pkl"):
    try:
        with open(ruta, "wb") as archivo:
            pickle.dump(modelo, archivo)
        print("\n✅ Modelo guardado exitosamente en 'modelo.pkl'.")
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")

# 🔥 **Ejecución del pipeline**
if __name__ == "__main__":
    print("🚀 Cargando datos...")
    df = cargar_datos()
    
    if df is not None:
        print("🔍 Preprocesando datos...")
        X_train, X_test, y_train, y_test = preprocesar_datos(df)

        print("🛠️ Entrenando modelo...")
        modelo = entrenar_modelo(X_train, y_train)

        print("📊 Evaluando modelo...")
        evaluar_modelo(modelo, X_test, y_test)

        print("🎯 Realizando predicción con un dato aleatorio...")
        prediccion_aleatoria(modelo, X_test, y_test)

        print("💾 Guardando modelo...")
        guardar_modelo(modelo)
