import streamlit as st
import pandas as pd
import plotly.express as px


def mostrar_vista_bi(df: pd.DataFrame) -> None:
    st.header("BI")
    st.write("Explorador flexible para cortes y descarga de pronósticos.")

    if df.empty:
        st.warning("No hay datos disponibles para mostrar.")
        return

    df_bi = df.copy()

    if "forecast" not in df_bi.columns and "item_cnt_month" in df_bi.columns:
        df_bi = df_bi.rename(columns={"item_cnt_month": "forecast"})

    columnas_requeridas = ["shop_id", "item_id", "forecast"]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_bi.columns]

    if columnas_faltantes:
        st.error(f"Faltan columnas necesarias: {columnas_faltantes}")
        return

    df_bi["forecast"] = pd.to_numeric(df_bi["forecast"], errors="coerce").fillna(0)

    columna_tienda = "shop_name" if "shop_name" in df_bi.columns else "shop_id"
    columna_categoria = (
        "item_category_name"
        if "item_category_name" in df_bi.columns
        else "item_category_id"
    )
    columna_producto = "item_name" if "item_name" in df_bi.columns else "item_id"

    if "forecast_year" in df_bi.columns and "forecast_month" in df_bi.columns:
        df_bi["temporada"] = (
            df_bi["forecast_year"].astype(str)
            + "-"
            + df_bi["forecast_month"].astype(int).astype(str).str.zfill(2)
        )
    else:
        df_bi["temporada"] = "Mes de predicción"

    if "precio_unitario_estimado" not in df_bi.columns:
        df_bi["precio_unitario_estimado"] = 100.0

    df_bi["venta_estimada"] = df_bi["forecast"] * df_bi["precio_unitario_estimado"]

    if "actual" in df_bi.columns:
        df_bi["actual"] = pd.to_numeric(df_bi["actual"], errors="coerce").fillna(0)
        df_bi["error_abs"] = (df_bi["actual"] - df_bi["forecast"]).abs()
    else:
        df_bi["actual"] = 0.0
        df_bi["error_abs"] = 0.0

    with st.container():
        st.subheader("Filtros BI")

        col1, col2, col3 = st.columns(3)

        with col1:
            categorias = st.multiselect(
                "Categoría",
                options=sorted(df_bi[columna_categoria].dropna().unique()),
                default=sorted(df_bi[columna_categoria].dropna().unique()),
                key="bi_categorias",
            )

        with col2:
            tiendas = st.multiselect(
                "Tienda",
                options=sorted(df_bi[columna_tienda].dropna().unique()),
                default=sorted(df_bi[columna_tienda].dropna().unique()),
                key="bi_tiendas",
            )

        with col3:
            temporadas = st.multiselect(
                "Temporada",
                options=sorted(df_bi["temporada"].dropna().unique()),
                default=sorted(df_bi["temporada"].dropna().unique()),
                key="bi_temporadas",
            )

    df_filtrado = df_bi[
        df_bi[columna_categoria].isin(categorias)
        & df_bi[columna_tienda].isin(tiendas)
        & df_bi["temporada"].isin(temporadas)
    ].copy()

    if df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    st.subheader("Configuración del corte")

    dimensiones_disponibles = [
        columna_tienda,
        columna_categoria,
        columna_producto,
        "temporada",
    ]

    dimensiones_disponibles = [
        col for col in dimensiones_disponibles if col in df_filtrado.columns
    ]

    metricas_disponibles = [
        "forecast",
        "venta_estimada",
        "actual",
        "error_abs",
    ]

    dimensiones = st.multiselect(
        "Agrupar por",
        options=dimensiones_disponibles,
        default=[columna_categoria],
        key="bi_dimensiones",
    )

    metricas = st.multiselect(
        "Métricas",
        options=metricas_disponibles,
        default=["forecast", "venta_estimada"],
        key="bi_metricas",
    )

    if not dimensiones or not metricas:
        st.warning("Selecciona al menos una dimensión y una métrica.")
        return

    df_corte = (
        df_filtrado
        .groupby(dimensiones, as_index=False)[metricas]
        .sum()
    )

    st.subheader("Gráfico dinámico BI")

    metrica_grafico = st.selectbox(
        "Métrica para graficar",
        options=metricas,
        key="bi_metrica_grafico",
    )

    dimension_x = st.selectbox(
        "Dimensión eje X",
        options=dimensiones,
        key="bi_dimension_x",
    )

    dimension_color = st.selectbox(
        "Dimensión de color",
        options=["Sin color"] + dimensiones,
        key="bi_dimension_color",
    )

    color = None if dimension_color == "Sin color" else dimension_color

    fig = px.bar(
        df_corte.sort_values(metrica_grafico, ascending=False).head(30),
        x=dimension_x,
        y=metrica_grafico,
        color=color,
        title=f"{metrica_grafico} por {dimension_x}",
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Tabla dinámica BI")
    st.dataframe(df_corte, width="stretch")

    csv = df_corte.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar corte BI",
        data=csv,
        file_name="corte_bi_forecast.csv",
        mime="text/csv",
    )