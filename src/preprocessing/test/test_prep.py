"""Pruebas unitarias para funciones del step de preprocessing."""

import numpy as np
import pandas as pd
import pytest

from preprocessing.prep import add_group_mean_lag, add_lags, add_shop_size_category


def _make_monthly_df() -> pd.DataFrame:
    """Crea un DataFrame mínimo de ventas mensuales para los tests."""
    return pd.DataFrame(
        {
            "date_block_num": [0, 0, 0, 1, 1, 1],
            "shop_id": [1, 2, 3, 1, 2, 3],
            "item_id": [10, 10, 10, 10, 10, 10],
            "item_cnt_month": [100.0, 50.0, 10.0, 110.0, 55.0, 12.0],
            "item_price_mean": [5.0, 5.0, 5.0, 5.5, 5.5, 5.5],
        }
    )


def test_add_lags_genera_columnas_correctas():
    """add_lags debe crear columnas con el patrón <col>_lag_<N> para cada lag."""
    tbl = _make_monthly_df()
    lags = [1, 2]
    resultado = add_lags(tbl, lags=lags, feature_cols=["item_cnt_month"])

    assert "item_cnt_month_lag_1" in resultado.columns
    assert "item_cnt_month_lag_2" in resultado.columns
    # No debe existir un lag no solicitado
    assert "item_cnt_month_lag_3" not in resultado.columns


def test_add_shop_size_category_clasifica_tiendas():
    """add_shop_size_category debe generar exactamente las 3 categorías válidas."""
    tbl = _make_monthly_df()
    resultado = add_shop_size_category(tbl)

    assert "shop_size" in resultado.columns
    categorias_validas = {"small", "average", "large"}
    categorias_obtenidas = set(resultado["shop_size"].unique())
    assert categorias_obtenidas.issubset(categorias_validas)


def test_add_group_mean_lag_genera_columna_correcta():
    """add_group_mean_lag debe agregar la columna de promedio con el nombre esperado."""
    tbl = _make_monthly_df()
    nombre_feature = "shop_cnt_mean_lag_1"
    resultado = add_group_mean_lag(
        tbl,
        group_cols=["shop_id"],
        target_col="item_cnt_month",
        lag=1,
        feature_name=nombre_feature,
    )

    assert nombre_feature in resultado.columns
    # El mes 0 no tiene historial previo → NaN en lag 1
    filas_mes_0 = resultado[resultado["date_block_num"] == 0]
    assert filas_mes_0[nombre_feature].isna().all()
