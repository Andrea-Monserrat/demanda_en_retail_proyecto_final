import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime


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
        col6.metric("Venta real simulada", f"{producto['actual']:,.2f}")
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
        nuevo_feedback = {
            "feedback_id": len(st.session_state.get("feedback_operativo", [])) + 1,
            "fecha_feedback": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "usuario": "usuario_demo",
            "shop_id": producto["shop_id"],
            "item_id": producto["item_id"],
            "producto": str(producto[columna_producto]),
            "tienda": str(producto[columna_tienda]),
            "categoria": str(producto[columna_categoria]),
            "temporada": temporada,
            "forecast": float(producto["forecast"]),
            "actual": None if pd.isna(producto["actual"]) else float(producto["actual"]),
            "tipo_problema": tipo_problema,
            "severidad": severidad,
            "observacion": observacion,
            "estado": "Pendiente de análisis ML",
        }

        if "feedback_operativo" not in st.session_state:
            st.session_state["feedback_operativo"] = []

        st.session_state["feedback_operativo"].append(nuevo_feedback)
        st.success("✅ Feedback guardado correctamente.")

    st.divider()

    st.subheader("Productos identificados con problemas")

    feedback_nuevo = st.session_state.get("feedback_operativo", [])

    if not feedback_nuevo:
        st.info("Aún no hay productos con feedback registrado en esta sesión.")
        return

    df_feedback = pd.DataFrame(feedback_nuevo)

    columnas_feedback = [
        "feedback_id",
        "fecha_feedback",
        "usuario",
        "shop_id",
        "item_id",
        "producto",
        "tienda",
        "categoria",
        "temporada",
        "forecast",
        "actual",
        "tipo_problema",
        "severidad",
        "observacion",
        "estado",
    ]

    st.dataframe(df_feedback[columnas_feedback], width="stretch")

    st.subheader("Alertas por producto")

    df_alertas = (
        df_feedback
        .groupby(["producto", "severidad"], as_index=False)
        .size()
        .rename(columns={"size": "cantidad_alertas"})
    )

    fig = px.bar(
        df_alertas,
        x="producto",
        y="cantidad_alertas",
        color="severidad",
        title="Productos con feedback registrado",
    )
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, width="stretch")