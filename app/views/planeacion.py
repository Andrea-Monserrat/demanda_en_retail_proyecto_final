import pandas as pd
import streamlit as st
import plotly.express as px


def mostrar_vista_planeacion(df: pd.DataFrame) -> None:
    st.header("Dirección de planeación")
    st.write("Planeación de demanda esperada por tienda, producto, categoría y temporada.")

    if df.empty:
        st.warning("No hay datos disponibles para mostrar.")
        return

    df_planeacion = df.copy()

    if "forecast" not in df_planeacion.columns and "item_cnt_month" in df_planeacion.columns:
        df_planeacion = df_planeacion.rename(columns={"item_cnt_month": "forecast"})

    columnas_requeridas = ["shop_id", "item_id", "forecast"]
    columnas_faltantes = [
        col for col in columnas_requeridas if col not in df_planeacion.columns
    ]

    if columnas_faltantes:
        st.error(f"Faltan columnas necesarias: {columnas_faltantes}")
        return

    df_planeacion["forecast"] = pd.to_numeric(
        df_planeacion["forecast"], errors="coerce"
    ).fillna(0)

    columna_tienda = "shop_name" if "shop_name" in df_planeacion.columns else "shop_id"
    columna_categoria = (
        "item_category_name"
        if "item_category_name" in df_planeacion.columns
        else "item_category_id"
    )
    columna_producto = "item_name" if "item_name" in df_planeacion.columns else "item_id"

    if "forecast_year" in df_planeacion.columns and "forecast_month" in df_planeacion.columns:
        df_planeacion["temporada"] = (
            df_planeacion["forecast_year"].astype(str)
            + "-"
            + df_planeacion["forecast_month"].astype(int).astype(str).str.zfill(2)
        )
    else:
        df_planeacion["temporada"] = "Mes de predicción"

    with st.container():
        st.subheader("Filtros de planeación")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tiendas = st.multiselect(
                "Tienda",
                options=sorted(df_planeacion[columna_tienda].dropna().unique()),
                default=sorted(df_planeacion[columna_tienda].dropna().unique()),
                key="planeacion_tiendas",
            )

        with col2:
            categorias = st.multiselect(
                "Categoría",
                options=sorted(df_planeacion[columna_categoria].dropna().unique()),
                default=sorted(df_planeacion[columna_categoria].dropna().unique()),
                key="planeacion_categorias",
            )

        with col3:
            productos = st.multiselect(
                "Producto",
                options=sorted(df_planeacion[columna_producto].dropna().unique()),
                default=[],
                key="planeacion_productos",
                help="Deja vacío para incluir todos los productos.",
            )

        with col4:
            temporadas = st.multiselect(
                "Temporada",
                options=sorted(df_planeacion["temporada"].dropna().unique()),
                default=sorted(df_planeacion["temporada"].dropna().unique()),
                key="planeacion_temporadas",
            )

    mask = (
        df_planeacion[columna_tienda].isin(tiendas)
        & df_planeacion[columna_categoria].isin(categorias)
        & df_planeacion["temporada"].isin(temporadas)
    )

    if productos:
        mask = mask & df_planeacion[columna_producto].isin(productos)

    df_filtrado = df_planeacion.loc[mask].copy()

    if df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    forecast_total = df_filtrado["forecast"].sum()
    forecast_promedio = df_filtrado["forecast"].mean()
    productos_unicos = df_filtrado["item_id"].nunique()
    tiendas_unicas = df_filtrado["shop_id"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Forecast total", f"{forecast_total:,.0f}")
    col2.metric("Forecast promedio", f"{forecast_promedio:,.2f}")
    col3.metric("Productos únicos", f"{productos_unicos:,}")
    col4.metric("Tiendas únicas", f"{tiendas_unicas:,}")

    if "actual" in df_filtrado.columns:
        df_filtrado["actual"] = pd.to_numeric(
            df_filtrado["actual"], errors="coerce"
        ).fillna(0)

        actual_total = df_filtrado["actual"].sum()
        variacion = (
            (forecast_total - actual_total) / actual_total
            if actual_total > 0
            else 0
        )

        col1, col2 = st.columns(2)
        col1.metric("Venta real simulada", f"{actual_total:,.0f}")
        col2.metric(
            "Variación forecast vs real",
            f"{variacion:.2%}",
            delta=f"{variacion:.2%}",
        )

    st.subheader("Forecast por categoría")

    df_categoria = (
        df_filtrado.groupby(columna_categoria, as_index=False)
        .agg(
            forecast_total=("forecast", "sum"),
            productos=("item_id", "nunique"),
            tiendas=("shop_id", "nunique"),
        )
        .sort_values("forecast_total", ascending=False)
    )

    fig_categoria = px.bar(
        df_categoria.head(20),
        x=columna_categoria,
        y="forecast_total",
        title="Top categorías por demanda esperada",
    )
    fig_categoria.update_xaxes(tickangle=45)
    st.plotly_chart(fig_categoria, width="stretch")

    st.subheader("Forecast por tienda")

    df_tienda = (
        df_filtrado.groupby(columna_tienda, as_index=False)
        .agg(
            forecast_total=("forecast", "sum"),
            productos=("item_id", "nunique"),
        )
        .sort_values("forecast_total", ascending=False)
    )

    fig_tienda = px.bar(
        df_tienda.head(20),
        x=columna_tienda,
        y="forecast_total",
        title="Top tiendas por demanda esperada",
    )
    fig_tienda.update_xaxes(tickangle=45)
    st.plotly_chart(fig_tienda, width="stretch")

    st.subheader("Detalle para planeación")

    columnas_reporte = [
        "shop_id",
        "item_id",
        columna_tienda,
        columna_categoria,
        columna_producto,
        "temporada",
        "forecast",
    ]

    columnas_reporte = [
        col for col in columnas_reporte if col in df_filtrado.columns
    ]

    df_reporte = (
        df_filtrado[columnas_reporte]
        .sort_values("forecast", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(df_reporte, width="stretch")

    csv = df_reporte.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar reporte CSV",
        data=csv,
        file_name="reporte_planeacion.csv",
        mime="text/csv",
    )