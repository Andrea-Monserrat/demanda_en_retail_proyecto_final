import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np



def mostrar_vista_general(df: pd.DataFrame) -> None:
    st.header("Análisis general")
    st.write("Vista ejecutiva del pronóstico mensual de demanda.")


    if df.empty:
        st.warning("No hay datos disponibles para mostrar.")
        return

    df_pred = df.copy()

    # Nombre estándar para trabajar en la app
    if "forecast" not in df_pred.columns and "item_cnt_month" in df_pred.columns:
        df_pred = df_pred.rename(columns={"item_cnt_month": "forecast"})

    columnas_requeridas = ["shop_id", "item_id", "forecast"]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_pred.columns]

    if columnas_faltantes:
        st.error(f"Faltan columnas necesarias: {columnas_faltantes}")
        return

    df_pred["forecast"] = pd.to_numeric(df_pred["forecast"], errors="coerce").fillna(0)


    if "actual" in df_pred.columns:
        st.subheader("Evaluación del modelo simulada")

        df_eval = df_pred.copy()
        df_eval["actual"] = pd.to_numeric(df_eval["actual"], errors="coerce").fillna(0)

        df_eval["error_abs"] = (df_eval["actual"] - df_eval["forecast"]).abs()
        df_eval["error_pct"] = df_eval["error_abs"] / df_eval["actual"].replace(0, np.nan)
        df_eval["sesgo"] = df_eval["forecast"] - df_eval["actual"]

        mae = df_eval["error_abs"].mean()
        mape = df_eval["error_pct"].mean()

        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mae:,.2f}")
        col2.metric("MAPE", f"{mape:.2%}")

    total_predicho = df_pred["forecast"].sum()
    forecast_promedio = df_pred["forecast"].mean()
    productos = df_pred["item_id"].nunique()
    tiendas = df_pred["shop_id"].nunique()

    st.info(
        "Esta vista muestra el volumen esperado de demanda para el mes de predicción. "
        "Las métricas se calculan a partir de las predicciones generadas por el modelo."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demanda total predicha", f"{total_predicho:,.0f}")
    col2.metric("Forecast promedio", f"{forecast_promedio:,.2f}")
    col3.metric("Productos únicos", f"{productos:,}")
    col4.metric("Tiendas únicas", f"{tiendas:,}")

    st.subheader("Distribución del forecast")

    fig_hist = px.histogram(
        df_pred,
        x="forecast",
        nbins=50,
        title="Distribución de predicciones",
    )
    st.plotly_chart(fig_hist, width="stretch")

    st.subheader("Top categorías por demanda esperada")

    if "item_category_name" in df_pred.columns:
        df_categoria = (
            df_pred.groupby("item_category_name", as_index=False)
            .agg(
                forecast_total=("forecast", "sum"),
                productos=("item_id", "nunique"),
                tiendas=("shop_id", "nunique"),
            )
            .sort_values("forecast_total", ascending=False)
            .head(15)
        )

        fig_categoria = px.bar(
            df_categoria,
            x="item_category_name",
            y="forecast_total",
            title="Top 15 categorías por forecast total",
        )
        fig_categoria.update_xaxes(tickangle=45)
        st.plotly_chart(fig_categoria, width="stretch")
    else:
        st.warning("No se encontró `item_category_name`; se omitió el análisis por categoría.")

    st.subheader("Top tiendas por demanda esperada")

    columna_tienda = "shop_name" if "shop_name" in df_pred.columns else "shop_id"

    df_tienda = (
        df_pred.groupby(columna_tienda, as_index=False)
        .agg(
            forecast_total=("forecast", "sum"),
            productos=("item_id", "nunique"),
        )
        .sort_values("forecast_total", ascending=False)
        .head(15)
    )

    fig_tienda = px.bar(
        df_tienda,
        x=columna_tienda,
        y="forecast_total",
        title="Top 15 tiendas por forecast total",
    )
    fig_tienda.update_xaxes(tickangle=45)
    st.plotly_chart(fig_tienda, width="stretch")

    st.subheader("Productos con mayor demanda esperada")

    columnas_producto = ["item_id"]
    if "item_name" in df_pred.columns:
        columnas_producto.append("item_name")
    if "item_category_name" in df_pred.columns:
        columnas_producto.append("item_category_name")

    df_producto = (
        df_pred.groupby(columnas_producto, as_index=False)
        .agg(
            forecast_total=("forecast", "sum"),
            tiendas=("shop_id", "nunique"),
        )
        .sort_values("forecast_total", ascending=False)
        .head(50)
    )

    st.dataframe(df_producto, width="stretch")

    csv = df_producto.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar top productos",
        data=csv,
        file_name="top_productos_forecast.csv",
        mime="text/csv",
    )