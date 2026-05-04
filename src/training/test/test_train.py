"""Pruebas unitarias para funciones del step de training."""

import numpy as np
import pandas as pd

from training.train import ParticionEntrenamiento, _split_train_valid, rmse


def test_rmse_con_prediccion_perfecta():
    """RMSE debe ser 0.0 cuando la predicción coincide exactamente con el target."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y_true, y_pred) == 0.0


def test_rmse_calcula_valor_conocido():
    """RMSE con error constante de 1 en cada observación debe ser exactamente 1.0."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    resultado = rmse(y_true, y_pred)
    assert abs(resultado - 1.0) < 1e-9


def test_split_train_valid_separa_correctamente_por_mes():
    """_split_train_valid debe separar filas: train < valid_month, valid == valid_month."""
    matrix = pd.DataFrame(
        {
            "date_block_num": [0, 1, 2, 2, 3],
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "item_cnt_month": [10, 20, 30, 40, 50],
        }
    )
    feature_cols = ["feat1"]
    particion = _split_train_valid(matrix, feature_cols, valid_month=2)

    assert isinstance(particion, ParticionEntrenamiento)
    assert len(particion.x_train) == 2  # meses 0 y 1
    assert len(particion.x_valid) == 2  # mes 2 (dos filas)
    assert list(particion.x_train.columns) == feature_cols
