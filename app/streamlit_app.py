import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Retail Sales Forecasting")
st.write("Primera versión local con datos dummy.")


@st.cache_data
def cargar_datos_dummy() -> pd.DataFrame:
    fechas = pd.date_range("2025-01-01", periods=120, freq="D")

    datos = pd.DataFrame({
        "fecha": np.tile(fechas, 3),
        "tienda": np.repeat(["Tienda A", "Tienda B", "Tienda C"], len(fechas)),
        "ventas": np.concatenate([
            np.random.normal(1000, 120, len(fechas)),
            np.random.normal(1400, 180, len(fechas)),
            np.random.normal(900, 100, len(fechas)),
        ]).round(2),
    })

    return datos


df = cargar_datos_dummy()

st.sidebar.header("Filtros")
tiendas = st.sidebar.multiselect(
    "Selecciona tienda",
    options=df["tienda"].unique(),
    default=df["tienda"].unique(),
)

df_filtrado = df[df["tienda"].isin(tiendas)]

ventas_totales = df_filtrado["ventas"].sum()
venta_promedio = df_filtrado["ventas"].mean()
dias = df_filtrado["fecha"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Ventas totales", f"${ventas_totales:,.0f}")
col2.metric("Venta promedio diaria", f"${venta_promedio:,.0f}")
col3.metric("Días analizados", dias)

st.subheader("Ventas por día")

ventas_diarias = (
    df_filtrado
    .groupby(["fecha", "tienda"], as_index=False)["ventas"]
    .sum()
)

fig = px.line(
    ventas_diarias,
    x="fecha",
    y="ventas",
    color="tienda",
    title="Evolución de ventas dummy",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Datos base")
st.dataframe(df_filtrado, use_container_width=True)