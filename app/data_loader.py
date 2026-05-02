"""Carga de datos para la app Streamlit: RDS primario, archivos locales fallback."""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import streamlit as st

from db import query

# ── Catálogos estáticos (lookup inmutable) ───────────────────────────────────
@st.cache_data
def cargar_catalogos():
    """Carga items, categorías y tiendas desde archivos locales."""
    items = pd.read_csv("data/raw/items.csv")
    cats = pd.read_csv("data/raw/item_categories.csv")
    shops = pd.read_csv("data/raw/shops.csv")
    return items, cats, shops


# ── Datos principales: predicciones + actuals ────────────────────────────────
@st.cache_data(ttl=300)
def cargar_datos_app() -> pd.DataFrame:
    """
    Devuelve un DataFrame con las columnas que esperan las vistas:
    item_id, shop_id, forecast, actual, lower_bound, upper_bound,
    item_name, item_category_id, item_category_name, shop_name,
    forecast_year, forecast_month, temporada.

    Intenta RDS primero; si falla, usa archivos locales (fallback).
    """
    try:
        return _cargar_desde_rds()
    except Exception as e:
        st.warning(f"RDS no disponible ({e}). Usando datos locales fallback.")
        return _cargar_desde_local()


def _cargar_desde_rds() -> pd.DataFrame:
    sql = """
    SELECT 
        pr.item_id,
        pr.shop_id,
        pred.predicted_sales   AS forecast,
        pred.lower_bound,
        pred.upper_bound,
        act.actual_sales       AS actual
    FROM predictions pred
    JOIN products pr ON pred.product_id = pr.id
    LEFT JOIN actuals act 
           ON act.product_id = pr.id 
          AND act.sale_date = '2015-10-01'
    WHERE pred.prediction_date = '2015-11-01'
    """
    rows = query(sql)
    if not rows:
        raise ValueError("No hay predicciones en RDS para 2015-11-01.")

    df = pd.DataFrame(rows)

    # Tipado numérico
    df["item_id"] = df["item_id"].astype(int)
    df["shop_id"] = df["shop_id"].astype(int)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce").fillna(0).clip(0, 20)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0)
    df["lower_bound"] = pd.to_numeric(df["lower_bound"], errors="coerce").fillna(0)
    df["upper_bound"] = pd.to_numeric(df["upper_bound"], errors="coerce").fillna(0)

    # Enriquecer con catálogos locales
    items, cats, shops = cargar_catalogos()
    df = df.merge(
        items[["item_id", "item_name", "item_category_id"]],
        on="item_id", how="left"
    )
    df = df.merge(
        cats[["item_category_id", "item_category_name"]],
        on="item_category_id", how="left"
    )
    df = df.merge(
        shops[["shop_id", "shop_name"]],
        on="shop_id", how="left"
    )

    # Compatibilidad con vistas existentes
    df["forecast_year"] = 2015
    df["forecast_month"] = 11
    df["temporada"] = "2015-11"

    return df


def _cargar_desde_local() -> pd.DataFrame:
    """Fallback: lee predictions.parquet + catálogos + simula actuals."""
    pred = pd.read_parquet("data/predictions/predictions.parquet")
    items, cats, shops = cargar_catalogos()

    df = pred.merge(items, on="item_id", how="left")
    df = df.merge(cats, on="item_category_id", how="left")
    df = df.merge(shops, on="shop_id", how="left")

    # Simular actuals con ruido (solo para desarrollo sin RDS)
    rng = np.random.default_rng(42)
    ruido = rng.normal(loc=0, scale=0.35, size=len(df))
    df["actual"] = (df["item_cnt_month"] * (1 + ruido)).clip(0, 20)

    df = df.rename(columns={"item_cnt_month": "forecast"})
    df["forecast"] = df["forecast"].fillna(0).clip(0, 20)

    test_month = 34
    df["date_block_num"] = test_month
    df["forecast_month"] = test_month % 12
    df["forecast_year"] = 2013 + test_month // 12
    df["temporada"] = f"{df['forecast_year'].iloc[0]}-{int(df['forecast_month'].iloc[0]):02d}"

    # lower/upper bound simulados para compatibilidad
    df["lower_bound"] = df["forecast"] * 0.85
    df["upper_bound"] = df["forecast"] * 1.15

    return df


# ── Métricas de evaluación desde RDS ─────────────────────────────────────────
@st.cache_data(ttl=300)
def cargar_evaluation_metrics() -> pd.DataFrame:
    """Carga RMSE/MAE vs naive desde evaluation_metrics."""
    try:
        rows = query("""
            SELECT group_key, rmse, mae, naive_rmse
            FROM evaluation_metrics
            ORDER BY group_key
        """)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(
            columns=["group_key", "rmse", "mae", "naive_rmse"]
        )


# ── Feedback desde RDS ───────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def cargar_feedback_rds() -> pd.DataFrame:
    """Carga feedback de negocio con joins a products."""
    try:
        rows = query("""
            SELECT 
                bf.id,
                bf.product_id,
                pr.item_id,
                pr.shop_id,
                bf.sentiment,
                bf.observation,
                bf.created_by,
                bf.created_at
            FROM business_feedback bf
            JOIN products pr ON bf.product_id = pr.id
            ORDER BY bf.created_at DESC
        """)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(
            columns=["id", "product_id", "item_id", "shop_id",
                     "sentiment", "observation", "created_by", "created_at"]
        )


# ── Productos flagged desde RDS ──────────────────────────────────────────────
@st.cache_data(ttl=60)
def cargar_flagged_rds() -> pd.DataFrame:
    """Carga productos marcados para revisión."""
    try:
        rows = query("""
            SELECT 
                fp.id,
                fp.product_id,
                pr.item_id,
                pr.shop_id,
                fp.reason,
                fp.resolved,
                fp.created_at
            FROM flagged_products fp
            JOIN products pr ON fp.product_id = pr.id
            WHERE fp.resolved = FALSE
            ORDER BY fp.created_at DESC
        """)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(
            columns=["id", "product_id", "item_id", "shop_id",
                     "reason", "resolved", "created_at"]
        )
