import pandas as pd
import streamlit as st


def mostrar_vista_finanzas(df: pd.DataFrame) -> None:
    st.header("Dirección de finanzas")
    st.write("Generación de reporte CFO con pronósticos del mes siguiente.")

    if df.empty:
        st.warning("No hay datos disponibles para mostrar.")
        return

    df_finanzas = df.copy()

    if "forecast" not in df_finanzas.columns and "item_cnt_month" in df_finanzas.columns:
        df_finanzas = df_finanzas.rename(columns={"item_cnt_month": "forecast"})

    columnas_requeridas = ["shop_id", "item_id", "forecast"]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_finanzas.columns]

    if columnas_faltantes:
        st.error(f"Faltan columnas necesarias: {columnas_faltantes}")
        return

    df_finanzas["forecast"] = pd.to_numeric(df_finanzas["forecast"], errors="coerce").fillna(0)

    columna_tienda = "shop_name" if "shop_name" in df_finanzas.columns else "shop_id"
    columna_categoria = (
        "item_category_name"
        if "item_category_name" in df_finanzas.columns
        else "item_category_id"
    )
    columna_producto = "item_name" if "item_name" in df_finanzas.columns else "item_id"

    if "forecast_year" in df_finanzas.columns and "forecast_month" in df_finanzas.columns:
        df_finanzas["temporada"] = (
            df_finanzas["forecast_year"].astype(str)
            + "-"
            + df_finanzas["forecast_month"].astype(int).astype(str).str.zfill(2)
        )
    else:
        df_finanzas["temporada"] = "Mes de predicción"

    # Como aún no tienes precio/costo real en predictions.parquet, simulamos una métrica financiera.
    # Cuando tengas precio real, reemplaza precio_unitario_estimado por la columna real.
    if "precio_unitario_estimado" not in df_finanzas.columns:
        df_finanzas["precio_unitario_estimado"] = 100.0

    df_finanzas["venta_estimada"] = (
        df_finanzas["forecast"] * df_finanzas["precio_unitario_estimado"]
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
                categorias = sorted(df_finanzas[columna_categoria].dropna().unique())
            else:
                categorias = st.multiselect(
                    "Categoría",
                    options=sorted(df_finanzas[columna_categoria].dropna().unique()),
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
                tiendas = sorted(df_finanzas[columna_tienda].dropna().unique())
            else:
                tiendas = st.multiselect(
                    "Tienda",
                    options=sorted(df_finanzas[columna_tienda].dropna().unique()),
                    default=[],
                    key="finanzas_tiendas",
                )

    df_reporte = df_finanzas[
        df_finanzas[columna_categoria].isin(categorias)
        & df_finanzas[columna_tienda].isin(tiendas)
    ].copy()

    if df_reporte.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    venta_total_estimada = df_reporte["venta_estimada"].sum()
    unidades_estimadas = df_reporte["forecast"].sum()
    productos_unicos = df_reporte["item_id"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Venta estimada", f"${venta_total_estimada:,.0f}")
    col2.metric("Unidades estimadas", f"{unidades_estimadas:,.0f}")
    col3.metric("Productos únicos", f"{productos_unicos:,}")

    columnas_reporte = [
        "shop_id",
        "item_id",
        columna_tienda,
        columna_categoria,
        columna_producto,
        "temporada",
        "forecast",
        "precio_unitario_estimado",
        "venta_estimada",
    ]

    columnas_reporte = [col for col in columnas_reporte if col in df_reporte.columns]

    df_reporte = (
        df_reporte[columnas_reporte]
        .sort_values("venta_estimada", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("Preview del reporte CFO")
    st.dataframe(df_reporte, width="stretch")

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