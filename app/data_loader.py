from pathlib import Path
import json
import pandas as pd
import streamlit as st
import numpy as np




def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def cargar_datos_app(
    predictions_path: str = "data/predictions/predictions.parquet",
    items_path: str = "data/raw/items.csv",
    categories_path: str = "data/raw/item_categories.csv",
    shops_path: str = "data/raw/shops.csv",
    meta_path: str = "data/prep/meta.json",
) -> pd.DataFrame:
    pred = _read_parquet(predictions_path)
    items = _read_csv(items_path)
    cats = _read_csv(categories_path)
    shops = _read_csv(shops_path)
    meta = _read_json(meta_path)

    df = pred.merge(items, on="item_id", how="left")
    df = df.merge(cats, on="item_category_id", how="left")
    df = df.merge(shops, on="shop_id", how="left")
    df = agregar_actual_simulado(df)

    test_month = int(meta["test_month"])
    df["date_block_num"] = test_month
    df["forecast_month"] = test_month % 12
    df["forecast_year"] = 2013 + test_month // 12

    df = df.rename(columns={"item_cnt_month": "forecast"})

    df["forecast"] = df["forecast"].fillna(0).clip(0, 20)
    
    return df



def agregar_actual_simulado(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()

    ruido = rng.normal(loc=0, scale=0.35, size=len(df))
    df["actual"] = df["forecast"] * (1 + ruido)

    # Para evitar negativos y respetar rango del problema
    df["actual"] = df["actual"].clip(0, 20)

    return df