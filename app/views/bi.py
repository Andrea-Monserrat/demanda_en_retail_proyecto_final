import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def cargar_datos_dummy() -> pd.DataFrame:
    df = pd.read_csv("data/dummy_sales.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")
    df["Forecast_venta_estimada"] = df["Forecast"] * df["Costo_unitario_producto"]
    return df

def mostrar_vista_bi() -> None:
    st.header("BI")
    st.write("Explorador flexible para cortes y descarga de pronósticos.")

    df = cargar_datos_dummy()

    df["Forecast_venta_estimada"] = (
        df["Forecast"] * df["Costo_unitario_producto"]
    )

    with st.container():
        st.subheader("Filtros BI")

        col1, col2, col3 = st.columns(3)

        with col1:
            categorias = st.multiselect(
                "Categoría",
                options=sorted(df["Categoria"].dropna().unique()),
                default=sorted(df["Categoria"].dropna().unique()),
                key="bi_categorias",
            )

        with col2:
            tiendas = st.multiselect(
                "Tienda",
                options=sorted(df["Tienda"].dropna().unique()),
                default=sorted(df["Tienda"].dropna().unique()),
                key="bi_tiendas",
            )

        with col3:
            tipos = st.multiselect(
                "Tipo",
                options=sorted(df["Tipo"].dropna().unique()),
                default=sorted(df["Tipo"].dropna().unique()),
                key="bi_tipos",
            )

    df_filtrado = df[
        (df["Categoria"].isin(categorias)) &
        (df["Tienda"].isin(tiendas)) &
        (df["Tipo"].isin(tipos))
    ].copy()

    st.subheader("Configuración del corte")

    dimensiones_disponibles = [
        "Tienda",
        "Categoria",
        "item_category_name",
        "Temporada",
        "Tipo",
    ]

    metricas_disponibles = [
        "Forecast",
        "Piezas_vendidas",
        "Venta_real",
        "Forecast_venta_estimada",
    ]

    dimensiones = st.multiselect(
        "Agrupar por",
        options=dimensiones_disponibles,
        default=["Categoria", "Tipo"],
        key="bi_dimensiones",
    )

    metricas = st.multiselect(
        "Métricas",
        options=metricas_disponibles,
        default=["Forecast", "Venta_real"],
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
        df_corte,
        x=dimension_x,
        y=metrica_grafico,
        color=color,
        title=f"{metrica_grafico} por {dimension_x}",
    )

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