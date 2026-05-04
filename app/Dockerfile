FROM python:3.12-slim

# Instalar curl (requerido para el HEALTHCHECK de ECS) y build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (gestor de paquetes moderno)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copiar manifiestos de dependencias primero (cacheo de capa Docker)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copiar el resto del código
COPY . .

# Activar el entorno virtual para evitar que uv run re-sincronice en cada arranque
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_NO_SYNC=1

EXPOSE 8501

# HealthCheck: si falla, ECS marca el contenedor como unhealthy y lo reinicia
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Ejecutar Streamlit directamente desde el venv (más rápido, sin re-sincronización)
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
