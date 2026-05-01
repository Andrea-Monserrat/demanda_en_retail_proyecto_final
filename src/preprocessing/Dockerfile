# Dockerfile — Step: preprocessing
FROM python:3.12-slim

# Instalar dependencias de machine learning
RUN pip install --no-cache-dir \
    numpy==2.2.4 \
    pandas==3.0.1 \
    pyarrow

# Directorio de trabajo dentro del contenedor
WORKDIR /app


COPY ./src ./src
#COPY ./utils ./utils
COPY ./data ./data


# PYTHONPATH para que los imports funcionen (ej: from utils.logging_config import ...)
ENV PYTHONPATH=/app/src:/app

ENTRYPOINT ["python", "-m", "preprocessing"]
