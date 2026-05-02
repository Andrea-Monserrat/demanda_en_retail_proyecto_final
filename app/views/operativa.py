import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

from db import query, execute_returning_id
from data_loader import cargar_feedback_rds, cargar_flagged_rds


def _buscar_product_id(item_id: int, shop_id: int) -> str | None:
    """Busca el UUID de products en RDS por item_id + shop_id."""
    rows = query(
        "SELECT id FROM products WHERE item_id = %s AND shop_id = %s",
        (str(item_id), str(shop_id)),
    )
    return rows[0]["id"] if rows else None


def mostrar_vista_operativa(df: pd.DataFrame) -> None:
    st.header("Vista operativa")
    st.write("Captura de feedback del negocio sobre productos con comportamiento problemático.")

    if df.empty:
        st.warning("No hay datos disponibles para mostrar.")
        return

    df_productos = df.copy()

    if "forecast" not in df_productos.columns and "item_cnt_month" in df_productos.columns:
        df_productos = df_productos.rename(columns={"item_cnt_month": "forecast"})

    columnas_requeridas = ["shop_id", "item_id", "forecast"]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_productos.columns]

    if columnas_faltantes:
        st.error(f"Faltan columnas necesarias: {columnas_faltantes}")
        return

    df_productos["forecast"] = pd.to_numeric(df_productos["forecast"], errors="coerce").fillna(0)

    if "actual" not in df_productos.columns:
        df_productos["actual"] = pd.NA

    columna_producto = "item_name" if "item_name" in df_productos.columns else "item_id"
    columna_tienda = "shop_name" if "shop_name" in df_productos.columns else "shop_id"
    columna_categoria = (
        "item_category_name"
        if "item_category_name" in df_productos.columns
        else "item_category_id"
    )

    st.subheader("Captura de feedback")

    df_productos["producto_label"] = (
        df_productos["item_id"].astype(str)
        + " — "
        + df_productos[columna_producto].astype(str)
        + " | Tienda: "
        + df_productos[columna_tienda].astype(str)
    )

    producto_label = st.selectbox(
        "Selecciona producto-tienda",
        options=sorted(df_productos["producto_label"].dropna().unique()),
        key="operativa_producto_id",
    )

    producto = df_productos[df_productos["producto_label"] == producto_label].iloc[0]
    item_id = int(producto["item_id"])
    shop_id = int(producto["shop_id"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Producto", str(producto[columna_producto]))
    col2.metric("Categoría", str(producto[columna_categoria]))
    col3.metric("Tienda", str(producto[columna_tienda]))

    col4, col5, col6 = st.columns(3)

    temporada = (
        f"{int(producto['forecast_year'])}-{int(producto['forecast_month']):02d}"
        if "forecast_year" in df_productos.columns and "forecast_month" in df_productos.columns
        else "Mes de predicción"
    )

    col4.metric("Temporada", temporada)
    col5.metric("Forecast", f"{producto['forecast']:,.2f}")

    if pd.notna(producto["actual"]):
        col6.metric("Venta real", f"{producto['actual']:,.2f}")
    else:
        col6.metric("Venta real", "Sin dato")

    tipo_problema = st.selectbox(
        "Tipo de problema",
        options=[
            "Forecast sobreestimado",
            "Forecast subestimado",
            "Producto sin stock",
            "Promoción no considerada",
            "Cambio de precio",
            "Producto descontinuado",
            "Error de datos",
            "Otro",
        ],
        key="operativa_tipo_problema",
    )

    severidad = st.selectbox(
        "Severidad",
        options=["Baja", "Media", "Alta"],
        key="operativa_severidad",
    )

    observacion = st.text_area(
        "Observación del negocio",
        placeholder="Describe qué está pasando con este producto...",
        key="operativa_observacion",
    )

    if st.button("Guardar feedback", key="operativa_guardar_feedback"):
        product_id = _buscar_product_id(item_id, shop_id)
        if not product_id:
            st.error("No se encontró el producto en la base de datos RDS.")
            return

        # Sentiment: negativo si severidad Alta, positivo si Baja, neutro si Media
        sentiment = "neutro"
        if severidad == "Alta":
            sentiment = "negativo"
        elif severidad == "Baja":
            sentiment = "positivo"

        # Observación enriquecida con metadatos
        observacion_full = f"[{tipo_problema} | Severidad: {severidad}] {observacion}"

        try:
            feedback_id = execute_returning_id(
                """
                INSERT INTO business_feedback (product_id, sentiment, observation, created_by)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (product_id, sentiment, observacion_full, "usuario_demo"),
            )

            if sentiment == "negativo" and feedback_id:
                execute_returning_id(
                    """
                    INSERT INTO flagged_products (product_id, feedback_id, reason)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (product_id, feedback_id, observacion_full[:500]),
                )

            st.success("Feedback guardado en RDS correctamente.")
            st.cache_data.clear()  # Invalidar cache de feedback/flagged
        except Exception as e:
            st.error(f"Error al guardar en RDS: {e}")

    st.divider()

    # ── Feedback histórico desde RDS ────────────────────────────────────────
    st.subheader("Feedback registrado")
    df_feedback = cargar_feedback_rds()
    if not df_feedback.empty:
        st.dataframe(df_feedback, width="stretch", use_container_width=True)
    else:
        st.info("Aún no hay feedback registrado en la base de datos.")

    st.divider()

    # ── Productos flagged (problemas sin resolver) ──────────────────────────
    st.subheader("Productos con problemas identificados")
    df_flagged = cargar_flagged_rds()
    if not df_flagged.empty:
        st.dataframe(df_flagged, width="stretch", use_container_width=True)

        st.subheader("Alertas por severidad")
        # Contamos por shop_id como proxy (o podríamos enriquecer con categoría)
        df_alertas = (
            df_flagged.groupby("shop_id", as_index=False)
            .size()
            .rename(columns={"size": "cantidad_alertas"})
            .sort_values("cantidad_alertas", ascending=False)
            .head(20)
        )
        fig = px.bar(
            df_alertas,
            x="shop_id",
            y="cantidad_alertas",
            title="Productos flagged por tienda",
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No hay productos marcados para revisión.")
