"""utils.logging_config: Configuración de logging para scripts del pipeline.

Este módulo centraliza la creación de loggers para:
- Escribir logs a archivo en artifacts/logs
- Imprimir logs en consola
- Incluir hostname y timestamp en cada línea

Se usa desde scripts como: prep.py, train.py, inference.py
"""

import logging
import socket
from datetime import datetime, timezone
from pathlib import Path


def get_logger(
    script_name: str, log_dir: str = "artifacts/logs"
) -> logging.LoggerAdapter:
    """
    Crea y devuelve un logger con:
    - Un archivo por corrida (timestamp UTC)
    - Formato consistente en consola y archivo
    - hostname agregado a cada línea vía LoggerAdapter

    Args:
        script_name: Nombre lógico del script (ej. "prep", "train", "inference").
        log_dir: Directorio donde se guardan los logs.

    Returns:
        LoggerAdapter listo para usarse con .info/.warning/.error.
    """
    # Asegura el directorio de logs (no falla si ya existe)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Archivo único por ejecución (UTC para evitar ambigüedades)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{script_name}_{timestamp}.log"

    # Logger base (uno por script_name)
    base_logger = logging.getLogger(script_name)
    base_logger.setLevel(logging.INFO)

    # Evita duplicar logs si el root logger también tiene handlers
    base_logger.propagate = False

    # Solo agregamos handlers si aún no existen (evita duplicados en imports repetidos)
    if not base_logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

        base_logger.handlers = [
        h for h in base_logger.handlers if isinstance(h, logging.StreamHandler)
        ]

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        base_logger.addHandler(file_handler)
        base_logger.addHandler(stream_handler)

    # Inyecta hostname en cada log sin repetirlo en cada llamada
    return logging.LoggerAdapter(base_logger, {"hostname": socket.gethostname()})
