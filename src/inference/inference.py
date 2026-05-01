"""inference.py: Inferencia batch para predicción de demanda en retail.

Entradas:
- data/inference/test.csv
- artifacts/models/model.joblib
- artifacts/models/feature_cols.json

Salida:
- data/predictions/predictions.csv
"""
# pylint: disable=duplicate-code

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.logging_config import get_logger

# Logger estándar del proyecto
logger = get_logger("inference")

# Nombre del target final y límites de clipping recomendados por el benchmark
TARGET_COL = "item_cnt_month"
CLIP_MIN = 0
CLIP_MAX = 20


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para el script de inferencia."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", default="data/inference")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--prep-dir", default="data/prep")
    parser.add_argument("--pred-dir", default="data/predictions")
    parser.add_argument("--model-file", default="model.joblib")
    return parser.parse_args()


def _validar_archivo_existe(ruta: Path) -> None:
    """Valida que un archivo exista (falla temprano y con contexto)."""
    if not ruta.exists():
        logger.error(
            "action=validate_inputs status=failure missing_file=%s",
            ruta.name,
        )
        raise FileNotFoundError(ruta)


def _cargar_modelo_y_columnas(
    dir_modelos: Path, nombre_modelo: str
) -> tuple[Any, list[str]]:
    """Carga el modelo entrenado y la lista de columnas (features) esperadas."""
    # Archivos de salida del entrenamiento
    ruta_modelo = dir_modelos / nombre_modelo
    ruta_columnas = dir_modelos / "feature_cols.json"

    _validar_archivo_existe(ruta_modelo)
    _validar_archivo_existe(ruta_columnas)

    # joblib para modelos sklearn
    modelo = joblib.load(ruta_modelo)
    # Las features deben coincidir exactamente con train.py
    columnas_features = json.loads(ruta_columnas.read_text(encoding="utf-8"))

    logger.info(
        "action=load_model status=success model_file=%s n_features=%s",
        ruta_modelo.name,
        len(columnas_features),
    )
    return modelo, columnas_features


def _cargar_test_csv(dir_inferencia: Path) -> pd.DataFrame:
    """Carga test.csv, que contiene shop_id/item_id y (a veces) el ID de Kaggle."""
    ruta_test = dir_inferencia / "test.csv"
    _validar_archivo_existe(ruta_test)

    datos_test = pd.read_csv(ruta_test, encoding="utf-8", low_memory=False)
    logger.info(
        "action=load_test status=success test_rows=%s",
        f"{len(datos_test):,}",
    )
    return datos_test

def _cargar_filas_test(prep_dir: Path) -> tuple[pd.DataFrame, int]:
    """Carga matrix.parquet y filtra únicamente las filas del mes de test."""
    ruta_matrix = prep_dir / "matrix.parquet"
    ruta_meta = prep_dir / "meta.json"

    _validar_archivo_existe(ruta_matrix)
    _validar_archivo_existe(ruta_meta)

    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    test_month = int(meta["test_month"])

    logger.info(
        "action=read_matrix status=started file=%s test_month=%s",
        ruta_matrix.name,
        test_month,
    )

    matrix = pd.read_parquet(ruta_matrix)

    filas_test = matrix.loc[matrix["date_block_num"] == test_month].copy()

    logger.info(
        "action=load_test_features status=success rows=%s test_month=%s",
        f"{len(filas_test):,}",
        test_month,
    )

    return filas_test, test_month



def _preparar_features(
    modelo: Any, filas_test: pd.DataFrame, columnas_features: list[str]
) -> pd.DataFrame:
    """Valida columnas, revisa NaNs y arma la matriz X (x_test) para el modelo."""
    # Validación: que existan todas las columnas requeridas por el modelo
    columnas_faltantes = [
        col for col in columnas_features if col not in filas_test.columns
    ]
    if columnas_faltantes:
        logger.error(
            "action=validate_features status=failure missing_cols=%s missing_total=%s",
            columnas_faltantes[:10],
            len(columnas_faltantes),
        )
        mensaje = (
            "Faltan columnas en matrix para inferencia: "
            f"{columnas_faltantes[:10]} (total={len(columnas_faltantes)})"
        )
        raise ValueError(mensaje)

    # X para inferencia (features en el mismo orden que en training)
    x_test = filas_test[columnas_features]

    # Warning si quedan NaNs (debería ser 0 si el prep está bien)
    celdas_nan = int(x_test.isna().sum().sum())
    if celdas_nan > 0:
        logger.warning(
            "action=validate_features status=warning nan_cells=%s",
            f"{celdas_nan:,}",
        )

    # Si sklearn guardó nombres/orden de columnas, fuerza exactamente ese orden
    if hasattr(modelo, "feature_names_in_"):
        x_test = x_test.reindex(columns=list(modelo.feature_names_in_))
        logger.info("action=reindex_features status=success")

    return x_test


def _predecir(modelo: Any, x_test: pd.DataFrame) -> np.ndarray:
    """Predice y aplica clipping del target (0 a 20) por definición del problema."""
    # predict devuelve float; clip recorta a rango permitido por el benchmark
    preds = np.clip(modelo.predict(x_test), CLIP_MIN, CLIP_MAX)

    logger.info(
        "action=predict status=success n_predictions=%s",
        f"{len(preds):,}",
    )
    return preds


def _guardar_salida(
    datos_test: pd.DataFrame, filas_test: pd.DataFrame, preds: np.ndarray, dir_pred: Path
) -> Path:
    """Une predicciones con el test original y escribe predictions.csv."""
    # Construimos tabla mínima para merge por shop_id/item_id
    tabla_pred = filas_test[["shop_id", "item_id"]].copy()
    tabla_pred[TARGET_COL] = preds

    # Merge para respetar el orden/IDs del test original
    salida = datos_test.merge(tabla_pred, on=["shop_id", "item_id"], how="left")

    ruta_salida = dir_pred / "predictions.parquet"
    salida.to_parquet(ruta_salida, index=False)

    logger.info(
        "action=save status=success output_file=%s rows_out=%s",
        ruta_salida.name,
        f"{len(salida):,}",
    )
    return ruta_salida


def main() -> None:
    """Orquesta el flujo: cargar artefactos -> features -> predecir -> guardar."""
    args = parse_args()
    inicio = time.time()

    logger.info("action=inference status=started")

    # Directorios de entrada/salida
    dir_inferencia = Path(args.inference_dir)
    dir_modelos = Path(args.models_dir)
    prep_dir = Path(args.prep_dir)
    dir_pred = Path(args.pred_dir)
    dir_pred.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Cargar modelo y lista de features
        modelo, columnas_features = _cargar_modelo_y_columnas(
            dir_modelos, args.model_file
        )

        # 2) Cargar test.csv para conservar estructura/orden
        datos_test = _cargar_test_csv(dir_inferencia)

        # 3) Construir features del mes de test
        filas_test, test_month = _cargar_filas_test(prep_dir)

        # 4) Preparar X y validar consistencia de columnas
        x_test = _preparar_features(modelo, filas_test, columnas_features)

        # 5) Predecir
        preds = _predecir(modelo, x_test)

        # 6) Guardar predictions.csv
        ruta_salida = _guardar_salida(datos_test, filas_test, preds, dir_pred)

        # Logging de tiempo de ejecución
        duracion = time.time() - inicio
        logger.info(
            "action=inference status=success output_file=%s duration_seconds=%.2f",
            ruta_salida.name,
            duracion,
        )

        # Prints
        print("OK inference")
        print("Saved:", ruta_salida)

    except Exception as exc:
        # Log completo del stacktrace
        logger.error(
            "action=inference status=failure error_type=%s error_message=%s",
            type(exc).__name__,
            str(exc)[:200],
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    main()
