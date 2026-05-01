"""Pruebas unitarias para funciones del step de inference."""

import numpy as np
import pandas as pd

from inference.inference import _predecir


class _ModeloFake:
    """Modelo ficticio que devuelve valores predefinidos (fuera del rango [0, 20])."""

    def predict(self, x_test):  # pylint: disable=unused-argument
        """Retorna predicciones con valores por debajo y por encima del rango permitido."""
        return np.array([-5.0, 10.0, 25.0])


def test_predecir_aplica_clipping_min_y_max():
    """Las predicciones deben quedar recortadas al rango [0, 20]."""
    x_test = pd.DataFrame({"feat": [1, 2, 3]})
    preds = _predecir(_ModeloFake(), x_test)

    assert preds[0] == 0.0, "Valores negativos deben clippearse a 0"
    assert preds[1] == 10.0, "Valores dentro del rango no deben modificarse"
    assert preds[2] == 20.0, "Valores mayores a 20 deben clippearse a 20"
    assert len(preds) == 3
