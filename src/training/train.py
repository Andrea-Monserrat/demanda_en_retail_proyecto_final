"""train.py: Entrenamiento del modelo de predicción de demanda en retail.

Entradas:
- data/prep/matrix.csv.gz
- data/prep/feature_cols.json
- data/prep/meta.json

Salidas:
- artifacts/models/model.joblib
- artifacts/models/train_report.json
- artifacts/models/feature_cols.json
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_squared_error

from utils.logging_config import get_logger

logger = get_logger("train")

# Target final a predecir y baseline mínimo (lag 1)
TARGET_COL = "item_cnt_month"
BASELINE_COL = "item_cnt_month_lag_1"

# Clipping recomendado por la competencia
CLIP_MIN = 0
CLIP_MAX = 20

# Semilla para reproducibilidad
RANDOM_SEED = 42


@dataclass(frozen=True)
class ParticionEntrenamiento:
    """Contenedor simple para evitar demasiadas variables locales en main()."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_valid: pd.DataFrame
    y_valid: pd.Series


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_train_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para el entrenamiento."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-dir", default="data/prep")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--model-file", default="model.joblib")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Número de iteraciones del HistGradientBoosting",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Profundidad máxima del árbol en HistGradientBoosting",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
        help="Tasa de aprendizaje del HistGradientBoosting",
    )
    return parser.parse_args()


def _validar_archivo_existe(ruta: Path) -> None:
    """Valida existencia de un archivo para fallar temprano con contexto."""
    if not ruta.exists():
        logger.error("action=validate_inputs status=failure missing_file=%s", ruta.name)
        raise FileNotFoundError(ruta)


def _cargar_inputs_prep(
    prep_dir: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Carga matrix, feature_cols y meta generados por prep.py."""
    ruta_matrix = prep_dir / "matrix.parquet"
    ruta_feature_cols = prep_dir / "feature_cols.json"
    ruta_meta = prep_dir / "meta.json"

    _validar_archivo_existe(ruta_matrix)
    _validar_archivo_existe(ruta_feature_cols)
    _validar_archivo_existe(ruta_meta)

    logger.info("action=read_matrix status=started file=%s", ruta_matrix)
    matrix = pd.read_parquet(ruta_matrix)
    logger.info("action=read_matrix status=success rows=%s cols=%s", 
                len(matrix), 
                len(matrix.columns))

    feature_cols = json.loads(ruta_feature_cols.read_text(encoding="utf-8"))
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))

    logger.info(
        "action=load_prep status=success matrix_rows=%s n_features=%s "
        "files=(%s,%s,%s)",
        f"{len(matrix):,}",
        f"{len(feature_cols):,}",
        ruta_matrix.name,
        ruta_feature_cols.name,
        ruta_meta.name,
    )
    return matrix, feature_cols, meta


def _extraer_meses(meta: dict[str, Any]) -> tuple[int, int]:
    """Extrae valid_month y test_month desde meta.json."""
    valid_month = int(meta["valid_month"])
    test_month = int(meta["test_month"])

    logger.info(
        "action=months status=success valid_month=%s test_month=%s",
        valid_month,
        test_month,
    )
    return valid_month, test_month


def _split_train_valid(
    matrix: pd.DataFrame,
    feature_cols: list[str],
    valid_month: int,
) -> ParticionEntrenamiento:
    """Separa train/valid por mes de validación (valid_month)."""
    # Train: meses anteriores a valid_month
    # Valid: el mes exacto valid_month
    mask_train = matrix["date_block_num"] < valid_month
    mask_valid = matrix["date_block_num"] == valid_month
    if matrix.loc[matrix["date_block_num"] == valid_month, TARGET_COL].empty:
        raise ValueError(f"No hay datos para valid_month={valid_month}")

    x_train = matrix.loc[mask_train, feature_cols]
    y_train = matrix.loc[mask_train, TARGET_COL]

    x_valid = matrix.loc[mask_valid, feature_cols]
    y_valid = matrix.loc[mask_valid, TARGET_COL]

    logger.info(
        "action=split status=success valid_month=%s train_rows=%s valid_rows=%s",
        valid_month,
        f"{len(x_train):,}",
        f"{len(x_valid):,}",
    )

    return ParticionEntrenamiento(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
    )


def _calcular_baseline_rmse(particion: ParticionEntrenamiento) -> float:
    """Calcula baseline RMSE usando item_cnt_month_lag_1 como predicción directa."""
    if BASELINE_COL not in particion.x_valid.columns:
        logger.error(
            "action=validate_inputs status=failure missing_baseline_feature=%s",
            BASELINE_COL,
        )
        raise ValueError(
            f"Baseline feature '{BASELINE_COL}' no existe en feature_cols/prep output."
        )

    pred_baseline = particion.x_valid[BASELINE_COL].to_numpy()
    baseline_rmse = rmse(particion.y_valid.to_numpy(), pred_baseline)

    logger.info(
        "action=baseline status=success baseline_feature=%s baseline_rmse=%.4f",
        BASELINE_COL,
        baseline_rmse,
    )
    return baseline_rmse


def _definir_modelos_candidatos(
    n_estimators: int = 400,
    max_depth: int = 8,
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    """Define modelos candidatos a comparar.

    Args:
        n_estimators: Número de iteraciones del HistGradientBoosting.
        max_depth: Profundidad máxima del árbol.
        learning_rate: Tasa de aprendizaje del HistGradientBoosting.
    """
    return {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            random_state=RANDOM_SEED,
        ),
        #"Ridge": Ridge(alpha=1.0),
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
            )
        ,
        "PoissonRegressor": make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=1e-4, max_iter=300),
            ),
          #PoissonRegressor(alpha=1e-4, max_iter=200),
    }


def _evaluar_modelos(
    modelos: dict[str, Any],
    particion: ParticionEntrenamiento,
) -> dict[str, float]:
    """Entrena y evalúa cada modelo candidato en validación (RMSE)."""
    scores: dict[str, float] = {}

    for nombre, modelo in modelos.items():
        logger.info("action=model_fit status=started model=%s", nombre)

        # Entrenamiento
        modelo.fit(particion.x_train, particion.y_train)

        # Predicción y clipping
        pred = np.clip(modelo.predict(particion.x_valid), CLIP_MIN, CLIP_MAX)
        score = rmse(particion.y_valid.to_numpy(), pred)
        scores[nombre] = score

        logger.info(
            "action=model_fit status=success model=%s rmse=%.4f",
            nombre,
            score,
        )

    return scores


def _seleccionar_mejor_modelo(scores: dict[str, float]) -> str:
    """Selecciona el modelo con menor RMSE."""
    mejor_nombre = min(scores, key=scores.get)
    logger.info(
        "action=model_select status=success best_model=%s best_rmse=%.4f",
        mejor_nombre,
        scores[mejor_nombre],
    )
    return mejor_nombre


def _entrenar_modelo_final(
    modelo_base: Any,
    particion: ParticionEntrenamiento,
) -> Any:
    """Reentrena el mejor modelo usando train+valid para maximizar datos."""
    logger.info("action=fit_final status=started")
    x_full = pd.concat([particion.x_train, particion.x_valid], axis=0)
    y_full = pd.concat([particion.y_train, particion.y_valid], axis=0)

    logger.info(
        "action=fit_final_data status=success rows_full=%s cols=%s",
        f"{len(x_full):,}",
        x_full.shape[1],
    )

    # clone para mantener el objeto base limpio
    modelo_final = clone(modelo_base)
    logger.info("action=fit_final_model status=started")
    modelo_final.fit(x_full, y_full)
    logger.info("action=fit_final_model status=success")


    logger.info("action=fit_final status=success rows_full=%s", f"{len(x_full):,}")
    return modelo_final


def _construir_reporte(
    baseline_rmse: float,
    scores: dict[str, float],
    mejor_nombre: str,
    valid_month: int,
    test_month: int,
) -> dict[str, Any]:
    """Construye el JSON de reporte del entrenamiento."""
    return {
        "baseline_rmse_lag_1": baseline_rmse,
        "scores": scores,
        "best_model": mejor_nombre,
        "valid_month": valid_month,
        "test_month": test_month,
        "improvement_pct_vs_baseline": (
            (baseline_rmse - scores[mejor_nombre]) / baseline_rmse
            )
    }


def _guardar_artefactos(
    models_dir: Path,
    model_file: str,
    modelo_final: Any,
    report: dict[str, Any],
    feature_cols: list[str],
) -> None:
    """Guarda modelo y reportes en artifacts/models."""
    models_dir.mkdir(parents=True, exist_ok=True)

    ruta_modelo = models_dir / model_file
    ruta_report = models_dir / "train_report.json"
    ruta_cols = models_dir / "feature_cols.json"

    joblib.dump(modelo_final, ruta_modelo)
    ruta_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ruta_cols.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    logger.info(
        "action=save status=success saved_model=%s saved_report=%s saved_feature_cols=%s",
        ruta_modelo.name,
        ruta_report.name,
        ruta_cols.name,
    )


def _ejecutar_entrenamiento(
    prep_dir: Path,
    models_dir: Path,
    model_file: str,
    n_estimators: int = 400,
    max_depth: int = 8,
    learning_rate: float = 0.08,
) -> None:
    """Orquesta el pipeline de entrenamiento sin inflar variables locales en main()."""
    # 1) Cargar inputs del pipeline de prep
    matrix, feature_cols, meta = _cargar_inputs_prep(prep_dir)

    # 2) Meses desde meta
    valid_month, test_month = _extraer_meses(meta)

    # 3) Split train/valid
    particion = _split_train_valid(matrix, feature_cols, valid_month)

    # 4) Baseline con lag 1
    baseline_rmse = _calcular_baseline_rmse(particion)

    # 5) Evaluación de modelos candidatos
    candidatos = _definir_modelos_candidatos(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    scores = _evaluar_modelos(candidatos, particion)

    # 6) Selección y entrenamiento final
    mejor_nombre = _seleccionar_mejor_modelo(scores)
   #modelo_final = _entrenar_modelo_final(candidatos[mejor_nombre], particion)
    modelo_final = candidatos[mejor_nombre]
    logger.info("action=fit_final status=skipped reason=using_validated_model model=%s",
    mejor_nombre,
    )

    # 7) Guardar artefactos
    report = _construir_reporte(
        baseline_rmse=baseline_rmse,
        scores=scores,
        mejor_nombre=mejor_nombre,
        valid_month=valid_month,
        test_month=test_month,
    )
    _guardar_artefactos(
        models_dir=models_dir,
        model_file=model_file,
        modelo_final=modelo_final,
        report=report,
        feature_cols=feature_cols,
    )


def main() -> None:
    """Punto de entrada: parsea args, mide tiempo y maneja errores con contexto."""
    args = parse_train_args()
    tiempo_inicio = time.time()

    logger.info("action=train status=started")

    prep_dir = Path(args.prep_dir)
    models_dir = Path(args.models_dir)

    try:
        _ejecutar_entrenamiento(
            prep_dir=prep_dir,
            models_dir=models_dir,
            model_file=args.model_file,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )

        duracion = time.time() - tiempo_inicio
        logger.info("action=train status=success duration_seconds=%.2f", duracion)

    except FileNotFoundError as exc:
        logger.error(
            "action=train status=failure error_type=FileNotFoundError error_message=%s",
            str(exc)[:200],
            exc_info=True,
        )
        raise
    except Exception as exc:
        logger.error(
            "action=train status=failure error_type=%s error_message=%s",
            type(exc).__name__,
            str(exc)[:200],
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    main()
