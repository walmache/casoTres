# Usar una imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el script de Python y los archivos necesarios al contenedor
COPY preciosCasa.py /app/
COPY modelo.pkl /app/  
COPY requirements.txt /app/

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 5000 para que el contenedor escuche en ese puerto
EXPOSE 5000

# Comando para ejecutar el script de Python cuando inicie el contenedor
#CMD ["python", "preciosCasa.py"]
CMD ["python", "app.py"]
