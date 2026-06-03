# Usar una imagen base oficial de Python liviana
FROM python:3.10-slim

# Evitar que Python escriba archivos .pyc en disco y habilitar el búfer de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para OpenCV y compilación básica
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Crear un usuario no privilegiado para cumplir con las políticas de seguridad de Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Establecer el directorio de trabajo del usuario
WORKDIR $HOME/app

# Copiar el archivo de requerimientos primero para aprovechar la caché de Docker
COPY --chown=user requirements.txt* ./

# Instalar dependencias de Python
# Nota: Si el archivo requirements.txt no existe, se instalarán las dependencias por defecto.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --quiet \
    executorch \
    torch==2.11.0 \
    torchvision \
    numpy \
    opencv-python-headless \
    gradio \
    gradio_client

# Copiar el script del servidor y los modelos (.pte) al contenedor
COPY --chown=user app.py ./app.py
COPY --chown=user *.pte ./

# Exponer el puerto por defecto que usa Hugging Face Spaces (7860)
EXPOSE 7860

# Comando para ejecutar la aplicación de Gradio
CMD ["python", "app.py"]
