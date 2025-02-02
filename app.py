import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

# Leer la variable de entorno para obtener la ruta del modelo
MODEL_PATH = os.environ.get('MODEL_PATH', 'modelo.pkl')  # 'modelo.pkl' es el valor por defecto

# Cargar el modelo entrenado desde la ruta especificada
with open(MODEL_PATH, 'rb') as archivo:
    modelo = pickle.load(archivo)

# Crear la aplicación Flask
app = Flask(__name__)

# Endpoint para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del cuerpo de la solicitud (esperamos un JSON con el área de la casa)
    data = request.get_json()
    
    # Asegurarse de que el campo 'area' está presente
    if 'area' not in data:
        return jsonify({"error": "Falta el parámetro 'area' en la solicitud"}), 400
    
    # Obtener el área de la casa
    area = data['area']
    
    # Hacer la predicción
    prediccion = modelo.predict(np.array([[area]]))[0]
    
    # Redondear la predicción a 3 decimales
    prediccion_redondeada = round(prediccion, 3)
    
    # Retornar la predicción en formato JSON
    return jsonify({"prediccion_precio": prediccion_redondeada})

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
