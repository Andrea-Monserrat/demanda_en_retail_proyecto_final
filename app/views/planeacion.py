import pandas as pd
import streamlit as st
import plotly.express as px


@st.cache_data
def cargar_datos_dummy() -> pd.DataFrame:
    df = pd.read_csv("data/dummy_sales.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"],format="%d/%m/%Y",errors="coerce")

    return df


def mostrar_vista_planeacion() -> None:
    st.header("Dirección de planeación")

    df = cargar_datos_dummy()
    st.sidebar.header("Filtros")

    tiendas = st.sidebar.multiselect(
        "Selecciona tienda",
        options=sorted(df["Tienda"].dropna().unique()),
        default=sorted(df["Tienda"].dropna().unique()),
    )

    categorias = st.sidebar.multiselect(
        "Selecciona categoría",
        options=sorted(df["Categoria"].dropna().unique()),
        default=sorted(df["Categoria"].dropna().unique()),
    )

    producto = st.sidebar.multiselect(
        "Selecciona tipo de producto",
        options=sorted(df["item_category_name"].dropna().unique()),
        default=sorted(df["item_category_name"].dropna().unique()),
    )

    temporadas = st.sidebar.multiselect(
        "Selecciona temporada",
        options=sorted(df["Temporada"].dropna().unique()),
        default=sorted(df["Temporada"].dropna().unique()),
    )

    df_filtrado = df[
        (df["Tienda"].isin(tiendas)) &
        (df["Categoria"].isin(categorias)) &
        (df["item_category_name"].isin(producto)) &
        (df["Temporada"].isin(temporadas))
    ]

    df_hist = df_filtrado[df_filtrado["Tipo"] == "Historico"]
    df_forecast = df_filtrado[df_filtrado["Tipo"] == "Forecast"]
    venta_historica = df_hist["Venta_real"].sum()
    #(forecast próxima - venta temporada pasada) / venta temporada pasada
    forecast_total = df_forecast["Forecast"].sum()
    ventas_totales = (df_hist["Costo_unitario_producto"] * df_hist["Piezas_vendidas"]).sum()
    venta_promedio = df_filtrado["Costo_unitario_producto"].mean()

    orden_temporadas = ["1Q", "2Q", "3Q", "4Q"]
    temporada_actual = df_filtrado["Temporada"].iloc[0]
    idx = orden_temporadas.index(temporada_actual)
    temporada_anterior = orden_temporadas[idx - 1] if idx > 0 else "4Q"

    ventas_previas = df[(df["Temporada"] == temporada_anterior) &(df["Tipo"] == "Historico")]["Venta_real"].sum()
    if ventas_previas > 0:
        variacion = (forecast_total - ventas_previas) / ventas_previas
    else:
        variacion = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Ventas totales", f"${ventas_totales:,.0f}")
    col2.metric("Venta promedio", f"${venta_promedio:,.0f}")
    col3.metric("Venta histórica",f"${venta_historica:,.0f}")

    df_error = df_filtrado.dropna(subset=["Venta_real", "Forecast"])
    if not df_error.empty:
        error = (abs(df_error["Forecast"] - df_error["Venta_real"]) / df_error["Venta_real"]).mean()
    else:
        error = 0

    col1, col2 = st.columns(2)

    col1.metric("Forecast próxima temporada",f"${forecast_total:,.0f}")
    delta_color = "normal" if variacion >= 0 else "inverse"
    col2.metric("Variación vs temporada anterior",f"{variacion:.2%}",delta=f"{variacion:.2%}",delta_color=delta_color)
    #col4.metric("Error esperado del modelo",f"{error:.2%}")

    df_filtrado = df_filtrado.copy()
    df_filtrado["valor_visualizado"] = df_filtrado.apply(lambda row: row["Venta_real"] if row["Tipo"] == "Historico" else row["Forecast"],axis=1)

    df_reporte = (df_filtrado.groupby(["Fecha", "Tienda", "Categoria", "item_category_name", "Temporada", "Tipo"],as_index=False).agg(valor=("valor_visualizado", "sum")).sort_values("Fecha"))

    st.subheader("Planeación de ventas")

    fig = px.line(
        df_reporte,
        x="Fecha",
        y="valor",
        color="Tipo",
        line_dash="Tienda",
        facet_col="Categoria",
        title="Evolución de ventas"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Datos visualizados")
    st.dataframe(df_reporte, width="stretch")

    csv = df_reporte.to_csv(index=False).encode("utf-8")

    if st.download_button(
        label="Descargar reporte CSV",
        data=csv,
        file_name="reporte_planeacion.csv",
        mime="text/csv",
    ):
        st.success("✅ Descarga completada correctamente")