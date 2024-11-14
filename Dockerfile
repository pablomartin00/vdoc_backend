# Usar la imagen base de Python 3.11.9
FROM python:3.11.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /apptmp

# Copiar el archivo de requerimientos al contenedor (si existe en tu directorio local)
COPY requirements.txt .

# Instalar las dependencias listadas en el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido de tu directorio local al contenedor (excepto lo especificado en .dockerignore)
COPY . .

# Exponer el puerto 5000 para Flask
EXPOSE 5001
#
# Comando para ejecutar la aplicación Flask (ajusta según tu entrypoint)
#CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
CMD ["tail", "-f", "/dev/null"]