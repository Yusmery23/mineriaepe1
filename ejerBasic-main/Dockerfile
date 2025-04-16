# Imagen base oficial de Python
FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para matplotlib y fonts
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \  
    libpng-dev \
    libjpeg-dev \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements y luego instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Comando por defecto para ejecutar el script
CMD ["python", "main.py"]
