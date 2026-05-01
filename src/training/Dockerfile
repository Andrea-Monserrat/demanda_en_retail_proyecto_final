# Dockerfile — Step: training
FROM python:3.12-slim

# Instalar dependencias de machine learning
RUN pip install --no-cache-dir \
    joblib==1.5.3 \
    numpy==2.4.2 \
    pandas==3.0.1 \
    scikit-learn==1.8.0 \
    pyarrow

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar código fuente
COPY src/ ./src/

# PYTHONPATH para que los imports funcionen (ej: from utils.logging_config import ...)
ENV PYTHONPATH=/app/src:/app

ENTRYPOINT ["python", "-m", "training"]
