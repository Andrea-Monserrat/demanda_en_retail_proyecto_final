import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def cargar_datos_dummy() -> pd.DataFrame:
    df = pd.read_csv("data/dummy_sales.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")

    columnas_numericas = ["Venta_real", "Forecast", "Costo_unitario_producto", "Piezas_vendidas"]
    for columna in columnas_numericas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

    return df


def mostrar_vista_general() -> None:
    st.header("Análisis general")
    st.write("Validación ejecutiva del modelo: forecast vs ground truth.")

    df = cargar_datos_dummy()

    df_eval = df.dropna(subset=["Venta_real", "Forecast"]).copy()

    if df_eval.empty:
        st.warning("No hay datos suficientes para evaluar el modelo.")
        return

    df_eval["error_abs"] = (df_eval["Venta_real"] - df_eval["Forecast"]).abs()
    df_eval["error_pct"] = df_eval["error_abs"] / df_eval["Venta_real"]
    df_eval["sesgo"] = df_eval["Forecast"] - df_eval["Venta_real"]

    mae = df_eval["error_abs"].mean()
    mape = df_eval["error_pct"].mean()
    sesgo_promedio = df_eval["sesgo"].mean()
    productos_evaluados = df_eval["item_category_id"].nunique()

    st.info(
        "Esta vista compara las predicciones contra el ground truth observado. "
        "La línea diagonal del scatter representa una predicción perfecta."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE global", f"{mae:,.0f}")
    col2.metric("MAPE global", f"{mape:.2%}")
    col3.metric("Sesgo promedio", f"{sesgo_promedio:,.0f}")
    col4.metric("Productos evaluados", productos_evaluados)

    st.subheader("Venta real vs Forecast")

    fig_scatter = px.scatter(
        df_eval,
        x="Venta_real",
        y="Forecast",
        color="error_pct",
        hover_data=["item_category_name", "Tienda", "Categoria"],
        title="Venta real vs Forecast",
    )

    min_val = min(df_eval["Venta_real"].min(), df_eval["Forecast"].min())
    max_val = max(df_eval["Venta_real"].max(), df_eval["Forecast"].max())

    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Predicción perfecta",
            line=dict(dash="dash"),
        )
    )

    fig_scatter.update_xaxes(type="log")
    fig_scatter.update_yaxes(type="log")

    st.plotly_chart(fig_scatter, width="stretch")

    st.subheader("Evaluación temporal: Forecast vs Ground Truth")

    fig_line = px.line(
        df_eval.sort_values("Fecha"),
        x="Fecha",
        y=["Venta_real", "Forecast"],
        color="item_category_name",
        title="Forecast vs Ground Truth por fecha",
    )

    st.plotly_chart(fig_line, width="stretch")

    st.subheader("Error promedio por categoría")

    df_error_categoria = (
        df_eval
        .groupby("Categoria", as_index=False)
        .agg(
            mae=("error_abs", "mean"),
            mape=("error_pct", "mean"),
            venta_real=("Venta_real", "sum"),
            forecast=("Forecast", "sum"),
        )
        .sort_values("mape", ascending=False)
    )

    fig_categoria = px.bar(
        df_error_categoria,
        x="Categoria",
        y="mape",
        title="MAPE por categoría",
    )

    st.plotly_chart(fig_categoria, width="stretch")

    st.subheader("Error por producto")

    df_error_producto = (
        df_eval
        .groupby(["item_category_id", "item_category_name", "Categoria"], as_index=False)
        .agg(
            mae=("error_abs", "mean"),
            mape=("error_pct", "mean"),
            venta_real=("Venta_real", "sum"),
            forecast=("Forecast", "sum"),
            sesgo=("sesgo", "mean"),
        )
        .sort_values("mape", ascending=False)
    )

    st.dataframe(df_error_producto, width="stretch")

    csv = df_error_producto.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar evaluación por producto",
        data=csv,
        file_name="evaluacion_modelo_por_producto.csv",
        mime="text/csv",
    )