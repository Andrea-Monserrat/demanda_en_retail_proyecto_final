import pandas as pd
import streamlit as st


@st.cache_data
def cargar_datos_dummy() -> pd.DataFrame:
    df = pd.read_csv("data/dummy_sales.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")
    df["Forecast_venta_estimada"] = df["Forecast"] * df["Costo_unitario_producto"]
    return df


def mostrar_vista_finanzas() -> None:
    st.header("Dirección de finanzas")
    st.write("Generación de reporte CFO con pronósticos del mes siguiente.")

    df = cargar_datos_dummy()

    df_forecast = df[df["Tipo"] == "Forecast"].copy()

    df_forecast["Forecast_venta_estimada"] = (
        df_forecast["Forecast"] * df_forecast["Costo_unitario_producto"]
    )

    with st.container():
        st.subheader("Filtros de finanzas")

        col1, col2 = st.columns(2)

        with col1:
            seleccionar_todas_categorias = st.checkbox(
                "Seleccionar todas las categorías",
                value=True,
                key="finanzas_todas_categorias",
            )

            if seleccionar_todas_categorias:
                categorias = sorted(df_forecast["Categoria"].dropna().unique())
            else:
                categorias = st.multiselect(
                    "Categoría",
                    options=sorted(df_forecast["Categoria"].dropna().unique()),
                    default=[],
                    key="finanzas_categorias",
                )

        with col2:
            seleccionar_todas_tiendas = st.checkbox(
                "Seleccionar todas las tiendas",
                value=True,
                key="finanzas_todas_tiendas",
            )

            if seleccionar_todas_tiendas:
                tiendas = sorted(df_forecast["Tienda"].dropna().unique())
            else:
                tiendas = st.multiselect(
                    "Tienda",
                    options=sorted(df_forecast["Tienda"].dropna().unique()),
                    default=[],
                    key="finanzas_tiendas",
                )


    df_reporte = df_forecast[
        (df_forecast["Categoria"].isin(categorias)) &
        (df_forecast["Tienda"].isin(tiendas))
    ].copy()

    columnas_reporte = [
        "Fecha",
        "Tienda",
        "Categoria",
        "item_category_name",
        "Temporada",
        "Forecast",
        "Costo_unitario_producto",
        "Forecast_venta_estimada",
    ]

    df_reporte = df_reporte[columnas_reporte]

    st.subheader("Preview del reporte CFO")
    st.dataframe(df_reporte, width="stretch")

    if df_reporte.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    csv = df_reporte.to_csv(index=False).encode("utf-8")

    if st.button("Generar reporte CFO"):
        st.session_state["reporte_cfo_generado"] = True
        st.success("✅ Reporte generado correctamente.")
        st.info("El archivo está listo para descarga local.")

    if st.session_state.get("reporte_cfo_generado", False):
        st.download_button(
            label="Descargar reporte CFO",
            data=csv,
            file_name="reporte_cfo_forecast_mes_siguiente.csv",
            mime="text/csv",
        )