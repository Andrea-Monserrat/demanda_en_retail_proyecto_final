"""prep.py: Preprocesamiento y generación de features para el modelo de retail.

Entradas:
- data/raw/items.csv
- data/raw/sales_train.csv
- data/raw/test.csv

Salidas:
- data/prep/matrix.csv.gz
- data/prep/feature_cols.json
- data/prep/meta.json
- data/prep/test_pairs.csv
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("prep")

# Llaves del dataset a nivel mes-tienda-producto
KEY_COLS: list[str] = ["date_block_num", "shop_id", "item_id"]

# Variables base a las que se les generan rezagos
LAG_FEATURE_COLS: list[str] = ["item_cnt_month", "item_price_mean"]
DEFAULT_LAGS: list[int] = [1, 2, 3, 6, 12]

# Target a predecir y clipping recomendado por la competencia
TARGET_COL: str = "item_cnt_month"
CLIP_TARGET_MIN: int = 0
CLIP_TARGET_MAX: int = 20


class MetaDict(TypedDict):
    """Metadatos mínimos para reproducir y depurar el pipeline de features."""

    first_month: int
    last_month: int
    test_month: int
    valid_month: int
    lags: list[int]
    n_rows_matrix: int
    n_features: int


def add_shop_size_category(
    tbl: pd.DataFrame,
    shop_col: str = "shop_id",
    target_col: str = "item_cnt_month",
) -> pd.DataFrame:
    """Clasifica tiendas por tamaño usando percentiles del total vendido.

    - 'small'  : <= p33
    - 'average': entre p33 y p66
    - 'large'  : > p66
    """
    # Total vendido por tienda en todo el histórico
    totales_tienda = (
        tbl.groupby(shop_col)[target_col].sum().sort_values(ascending=False)
    )

    # Umbrales por percentil
    p33 = totales_tienda.quantile(0.33)
    p66 = totales_tienda.quantile(0.66)

    # Mapeo vectorizado de categoría
    shop_size_map = pd.Series("average", index=totales_tienda.index)
    shop_size_map.loc[totales_tienda <= p33] = "small"
    shop_size_map.loc[totales_tienda > p66] = "large"

    resultado = tbl.copy()
    resultado["shop_size"] = resultado[shop_col].map(shop_size_map)
    return resultado


def add_lags(
    tbl: pd.DataFrame,
    lags: Sequence[int],
    feature_cols: Sequence[str],
    key_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Agrega variables rezagadas (lag features) para `feature_cols`.

    Para cada lag:
    - Desplaza date_block_num a futuro (+lag)
    - Une por llaves (left join) para traer el valor pasado
    """
    if key_cols is None:
        key_cols = KEY_COLS

    resultado = tbl.copy()
    llaves = list(key_cols)
    features = list(feature_cols)

    for lag in lags:
        # Tabla auxiliar con llaves + features y mes desplazado
        tbl_lag = resultado[llaves + features].copy()
        tbl_lag["date_block_num"] = tbl_lag["date_block_num"] + lag

        # Renombrar columnas para indicar el rezago
        renombres = {col: f"{col}_lag_{lag}" for col in features}
        tbl_lag = tbl_lag.rename(columns=renombres)

        # Merge contra tabla principal
        resultado = resultado.merge(tbl_lag, on=llaves, how="left")

    return resultado


def add_group_mean_lag(
    tbl: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str,
    lag: int = 1,
    feature_name: str | None = None,
) -> pd.DataFrame:
    """Agrega promedio por grupo, con desplazamiento temporal y unión.

    Ejemplo:
    - promedio mensual por shop_id
    - después se desplaza un mes (+1) y se une al mes actual
    """
    if feature_name is None:
        columnas_grupo_str = "_".join(group_cols)
        feature_name = f"{columnas_grupo_str}_{target_col}_mean_lag_{lag}"

    merge_cols = ["date_block_num", *group_cols]

    # Promedio por mes y grupo
    tbl_mean = (
        tbl.groupby(merge_cols, as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: feature_name})
    )
    # Desplazamiento hacia el futuro para que el "pasado" caiga en el mes actual
    tbl_mean["date_block_num"] = tbl_mean["date_block_num"] + lag

    return tbl.merge(tbl_mean, on=merge_cols, how="left")


def _cargar_datos_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los CSVs base desde data/raw."""
    items = pd.read_csv(raw_dir / "items.csv", encoding="utf-8", low_memory=False)
    sales_train = pd.read_csv(
        raw_dir / "sales_train.csv", encoding="utf-8", low_memory=False
    )
    test = pd.read_csv(raw_dir / "test.csv", encoding="utf-8", low_memory=False)

    logger.info(
        "action=load_data status=success rows_items=%s rows_sales_train=%s rows_test=%s",
        f"{len(items):,}",
        f"{len(sales_train):,}",
        f"{len(test):,}",
    )
    return items, sales_train, test


def _limpiar_ventas(sales_train: pd.DataFrame) -> pd.DataFrame:
    """Convierte fechas y filtra registros inválidos (precio<=0 o cantidad diaria < 0)."""
    tbl_sales = sales_train.copy()

    # Parseo de fecha del formato original
    tbl_sales["date"] = pd.to_datetime(
        tbl_sales["date"], format="%d.%m.%Y", errors="coerce"
    )

    bad_dates = int(tbl_sales["date"].isna().sum())
    if bad_dates > 0:
        logger.warning("action=parse_dates status=warning bad_dates=%s", f"{bad_dates:,}")

    # Filtros básicos para evitar valores negativos o inválidos
    filas_antes = len(tbl_sales)
    tbl_sales = tbl_sales[
        (tbl_sales["item_price"] > 0) & (tbl_sales["item_cnt_day"] >= 0)
    ].copy()
    filas_despues = len(tbl_sales)

    if filas_despues < filas_antes:
        logger.info(
            "action=filter_sales status=success rows_before=%s rows_after=%s dropped=%s",
            f"{filas_antes:,}",
            f"{filas_despues:,}",
            f"{filas_antes - filas_despues:,}",
        )

    # Revenue diario para agregación mensual
    tbl_sales["revenue_day"] = tbl_sales["item_price"] * tbl_sales["item_cnt_day"]
    return tbl_sales


def _agregar_mensual(tbl_sales: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventas diarias a nivel mensual por (mes, tienda, producto)."""
    tbl_month_sales = tbl_sales.groupby(KEY_COLS, as_index=False).agg(
        item_cnt_month=("item_cnt_day", "sum"),
        item_price_mean=("item_price", "mean"),
        revenue_month=("revenue_day", "sum"),
    )
    # Feature adicional: tamaño de tienda (small/average/large)
    return add_shop_size_category(tbl_month_sales)


def _crear_grid(
    tbl_sales: pd.DataFrame, first_month: int, last_month: int
) -> pd.DataFrame:
    """Crea la malla (mes, tienda, producto) para todos los meses del histórico."""
    grid_arrays: list[np.ndarray] = []

    for month in range(first_month, last_month + 1):
        # Se toman combinaciones observadas en ese mes (evita full Cartesian enorme)
        tbl_mes = tbl_sales[tbl_sales["date_block_num"] == month]
        shop_ids = tbl_mes["shop_id"].unique()
        item_ids = tbl_mes["item_id"].unique()

        grid_arrays.append(
            np.array(list(product([month], shop_ids, item_ids)), dtype=np.int32)
        )

    return pd.DataFrame(np.vstack(grid_arrays), columns=KEY_COLS)


def _agregar_mes_test(tbl_matrix: pd.DataFrame, test: pd.DataFrame, test_month: int) ->pd.DataFrame:
    """Agrega las combinaciones shop-item del test como mes adicional (test_month)."""
    tbl_test_pairs = test[["shop_id", "item_id"]].copy()
    tbl_test_pairs["date_block_num"] = test_month
    return pd.concat([tbl_matrix, tbl_test_pairs[KEY_COLS]], ignore_index=True)


def _codificar_features_base(
    tbl_matrix: pd.DataFrame, items: pd.DataFrame
) -> pd.DataFrame:
    """Crea y tipa features base (shop_size, categoría, month/year) y rellena NaNs."""
    resultado = tbl_matrix.copy()

    # shop_size: string -> int (y fillna por seguridad en filas del test)
    shop_size_map = {"small": 0, "average": 1, "large": 2}
    resultado["shop_size"] = (
        resultado["shop_size"].fillna("average").map(shop_size_map).astype(np.int8)
    )

    # Relleno y tipeo de columnas agregadas
    for col in ["item_cnt_month", "item_price_mean", "revenue_month"]:
        resultado[col] = resultado[col].fillna(0).astype(np.float32)

    # item_category_id desde items.csv
    cat_map = items.set_index("item_id")["item_category_id"]
    resultado["item_category_id"] = resultado["item_id"].map(cat_map).astype(np.int16)

    # Clipping del target (compatibilidad con la métrica/competencia)
    resultado[TARGET_COL] = resultado[TARGET_COL].clip(CLIP_TARGET_MIN, CLIP_TARGET_MAX)

    # Features temporales
    resultado["month"] = (resultado["date_block_num"] % 12).astype(np.int8)
    resultado["year"] = (2013 + resultado["date_block_num"] // 12).astype(np.int16)

    return resultado


def _eliminar_lags_previos(tbl_matrix: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas de lags si ya existen, para evitar duplicados."""
    cols_old = [c for c in tbl_matrix.columns if "lag_" in c] + [
        "shop_mean_lag_1",
        "item_mean_lag_1",
        "cat_mean_lag_1",
    ]
    return tbl_matrix.drop(columns=cols_old, errors="ignore")


def build_matrix(raw_dir: Path) -> tuple[pd.DataFrame, list[str], MetaDict, pd.DataFrame]:
    """Construye la matriz final de features y retorna artefactos necesarios.

    Retorna:
    - tbl_matrix: dataset final con features
    - feature_cols: lista de columnas de entrada al modelo
    - meta: metadatos del pipeline
    - tbl_test_pairs_with_id: pares con ID (útil para submission/inferencia)
    """
    logger.info("action=build_matrix status=started")

    # 1) Carga de datos
    items, sales_train, test = _cargar_datos_raw(raw_dir)

    # Guardamos pares del test con ID para reproducir salida estilo Kaggle
    tbl_test_pairs_with_id = test[["ID", "shop_id", "item_id"]].copy()

    # 2) Limpieza y agregación mensual
    tbl_sales = _limpiar_ventas(sales_train)

    first_month = int(tbl_sales["date_block_num"].min())
    last_month = int(tbl_sales["date_block_num"].max())
    test_month = last_month + 1

    logger.info(
        "action=month_range status=success first_month=%s last_month=%s test_month=%s",
        first_month,
        last_month,
        test_month,
    )

    tbl_month_sales = _agregar_mensual(tbl_sales)

    # 3) Construcción de la malla histórica + mes de test
    tbl_matrix = _crear_grid(tbl_sales, first_month, last_month)
    tbl_matrix = _agregar_mes_test(tbl_matrix, test, test_month)

    # 4) Merge con agregados mensuales
    tbl_matrix = tbl_matrix.merge(tbl_month_sales, on=KEY_COLS, how="left")

    # 5) Features base + tipado
    tbl_matrix = _codificar_features_base(tbl_matrix, items)

    # 6) Lags (rezagos) y promedios por grupo (lag 1)
    tbl_matrix = _eliminar_lags_previos(tbl_matrix)

    lags = DEFAULT_LAGS
    tbl_matrix = add_lags(tbl_matrix, lags, feature_cols=LAG_FEATURE_COLS)

    tbl_matrix = add_group_mean_lag(
        tbl_matrix,
        ["shop_id"],
        TARGET_COL,
        lag=1,
        feature_name="shop_mean_lag_1",
    )
    tbl_matrix = add_group_mean_lag(
        tbl_matrix,
        ["item_id"],
        TARGET_COL,
        lag=1,
        feature_name="item_mean_lag_1",
    )
    tbl_matrix = add_group_mean_lag(
        tbl_matrix,
        ["item_category_id"],
        TARGET_COL,
        lag=1,
        feature_name="cat_mean_lag_1",
    )

    # 7) Relleno final de NaNs en columnas de lags/medias
    lag_cols = [c for c in tbl_matrix.columns if "lag_" in c] + [
        "shop_mean_lag_1",
        "item_mean_lag_1",
        "cat_mean_lag_1",
    ]
    # Dedup preservando orden
    lag_cols = list(dict.fromkeys(lag_cols))
    tbl_matrix[lag_cols] = tbl_matrix[lag_cols].fillna(0).astype(np.float32)

    # 8) Lista final de features para el modelo
    feature_cols = [
        "shop_id",
        "item_id",
        "item_category_id",
        "month",
        "year",
        "shop_size",
    ] + lag_cols

    meta: MetaDict = {
        "first_month": first_month,
        "last_month": last_month,
        "test_month": test_month,
        "valid_month": last_month,
        "lags": list(lags),
        "n_rows_matrix": int(tbl_matrix.shape[0]),
        "n_features": int(len(feature_cols)),
    }

    logger.info(
        "action=build_matrix status=success rows_matrix=%s n_features=%s",
        f"{tbl_matrix.shape[0]:,}",
        f"{len(feature_cols):,}",
    )

    return tbl_matrix, feature_cols, meta, tbl_test_pairs_with_id


def parse_prep_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para el script de prep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="Directorio de entrada (raw)")
    parser.add_argument("--prep-dir", default="data/prep", help="Directorio de salida (prep)")
    return parser.parse_args()


def main() -> None:
    """Ejecuta el pipeline de preparación y escribe artefactos en data/prep."""
    args = parse_prep_args()
    start_time = time.time()

    logger.info("action=prep status=started")

    raw_dir = Path(args.raw_dir)
    prep_dir = Path(args.prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    try:
        matrix, feature_cols, meta, test_pairs_with_id = build_matrix(raw_dir)
    except Exception as exc:
        logger.error(
            "action=prep status=failure error_type=%s error_message=%s",
            type(exc).__name__,
            str(exc)[:200],
            exc_info=True,
        )
        raise

    # Guardar matriz comprimida (más ligera para git/transferencia local)
    out_matrix = prep_dir / "matrix.csv.gz"
    matrix.to_csv(out_matrix, index=False, compression="gzip")

    # Artefactos de apoyo: columnas, meta y pares del test
    (prep_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2),
        encoding="utf-8",
    )
    (prep_dir / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    test_pairs_with_id.to_csv(prep_dir / "test_pairs.csv", index=False)

    duration = time.time() - start_time
    logger.info(
        "action=prep status=success matrix_file=%s duration_seconds=%.2f",
        out_matrix.name,
        duration,
    )


if __name__ == "__main__":
    main()
