import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

from data_loader import cargar_evaluation_metrics


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

    # ── Métricas reales desde RDS ───────────────────────────────────────────
    df_metrics = cargar_evaluation_metrics()
    if not df_metrics.empty:
        st.subheader("Evaluación del modelo vs baseline naive")

        # KPI global
        global_row = df_metrics[df_metrics["group_key"] == "all"]
        if not global_row.empty:
            rmse_model = float(global_row["rmse"].iloc[0])
            rmse_naive = float(global_row["naive_rmse"].iloc[0])
            mejora = (rmse_naive - rmse_model) / rmse_naive if rmse_naive > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE modelo", f"{rmse_model:.4f}")
            col2.metric("RMSE naive", f"{rmse_naive:.4f}")
            col3.metric("Mejora vs naive", f"{mejora:.2%}")
        else:
            st.info("Métricas globales aún no calculadas.")

        # Tabla por categoría
        df_cat = df_metrics[df_metrics["group_key"].str.startswith("category:")].copy()
        if not df_cat.empty:
            df_cat["category"] = df_cat["group_key"].str.replace("category:", "", regex=False)
            df_cat["rmse_model"] = df_cat["rmse"].astype(float)
            df_cat["rmse_naive"] = df_cat["naive_rmse"].astype(float)
            df_cat["mejora"] = (df_cat["rmse_naive"] - df_cat["rmse_model"]) / df_cat["rmse_naive"]
            df_cat = df_cat.sort_values("mejora", ascending=False)

            st.subheader("RMSE por categoría")
            st.dataframe(
                df_cat[["category", "rmse_model", "rmse_naive", "mejora"]].style.format({
                    "rmse_model": "{:.4f}",
                    "rmse_naive": "{:.4f}",
                    "mejora": "{:.2%}",
                }),
                width="stretch",
            )

        # Tabla por tienda
        df_shop = df_metrics[df_metrics["group_key"].str.startswith("shop:")].copy()
        if not df_shop.empty:
            df_shop["shop"] = df_shop["group_key"].str.replace("shop:", "", regex=False)
            df_shop["rmse_model"] = df_shop["rmse"].astype(float)
            df_shop["rmse_naive"] = df_shop["naive_rmse"].astype(float)
            df_shop["mejora"] = (df_shop["rmse_naive"] - df_shop["rmse_model"]) / df_shop["rmse_naive"]
            df_shop = df_shop.sort_values("mejora", ascending=False)

            st.subheader("RMSE por tienda")
            st.dataframe(
                df_shop[["shop", "rmse_model", "rmse_naive", "mejora"]].style.format({
                    "rmse_model": "{:.4f}",
                    "rmse_naive": "{:.4f}",
                    "mejora": "{:.2%}",
                }),
                width="stretch",
            )
    else:
        st.info("Métricas de evaluación no disponibles en RDS.")

    # ── Evaluación con actuals (si existen) ─────────────────────────────────
    if "actual" in df_pred.columns and df_pred["actual"].notna().any():
        st.subheader("Scatter: predicción vs actual")
        df_eval = df_pred[df_pred["actual"] > 0].copy()
        fig = px.scatter(
            df_eval,
            x="actual",
            y="forecast",
            color="item_category_name" if "item_category_name" in df_eval.columns else None,
            opacity=0.6,
            title="Forecast vs Actual",
        )
        max_val = max(df_eval["actual"].max(), df_eval["forecast"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, width="stretch")

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