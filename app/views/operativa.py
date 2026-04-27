import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime


@st.cache_data
def cargar_datos_productos() -> pd.DataFrame:
    df = pd.read_csv("data/dummy_sales.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")
    return df


@st.cache_data
def cargar_feedback_dummy() -> pd.DataFrame:
    df = pd.read_csv("data/dummy_feedback.csv")
    df["fecha_feedback"] = pd.to_datetime(df["fecha_feedback"], errors="coerce")
    return df


def mostrar_vista_operativa() -> None:
    st.header("Vista operativa")
    st.write("Captura de feedback del negocio sobre productos con comportamiento problemático.")

    df_productos = cargar_datos_productos()
    df_feedback_base = cargar_feedback_dummy()

    st.subheader("Captura de feedback")

    producto_id = st.selectbox(
        "Selecciona producto",
        options=sorted(df_productos["item_category_id"].dropna().unique()),
        key="operativa_producto_id",
    )

    producto = df_productos[df_productos["item_category_id"] == producto_id].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Producto", producto["item_category_name"])
    col2.metric("Categoría", producto["Categoria"])
    col3.metric("Tienda", producto["Tienda"])

    col4, col5, col6 = st.columns(3)

    col4.metric("Temporada", producto["Temporada"])
    col5.metric("Forecast", f"{producto['Forecast']:,.0f}")
    col6.metric(
        "Venta real",
        f"${producto['Venta_real']:,.0f}" if pd.notna(producto["Venta_real"]) else "Sin dato",
    )

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
            "feedback_id": len(df_feedback_base) + len(st.session_state.get("feedback_operativo", [])) + 1,
            "fecha_feedback": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "usuario": "usuario_demo",
            "item_category_id": producto["item_category_id"],
            "item_category_name": producto["item_category_name"],
            "tienda": producto["Tienda"],
            "categoria": producto["Categoria"],
            "temporada": producto["Temporada"],
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

    if feedback_nuevo:
        df_feedback_nuevo = pd.DataFrame(feedback_nuevo)
        df_feedback = pd.concat([df_feedback_base, df_feedback_nuevo], ignore_index=True)
    else:
        df_feedback = df_feedback_base.copy()

    if df_feedback.empty:
        st.info("Aún no hay productos con feedback registrado.")
        return

    columnas_feedback = [
        "feedback_id",
        "fecha_feedback",
        "usuario",
        "item_category_id",
        "item_category_name",
        "tienda",
        "categoria",
        "temporada",
        "tipo_problema",
        "severidad",
        "observacion",
        "estado",
    ]

    df_feedback = df_feedback[columnas_feedback]

    st.dataframe(df_feedback, width="stretch")

    st.subheader("Alertas por producto")

    df_alertas = (
        df_feedback
        .groupby(["item_category_name", "severidad"], as_index=False)
        .size()
        .rename(columns={"size": "cantidad_alertas"})
    )

    fig = px.bar(
        df_alertas,
        x="item_category_name",
        y="cantidad_alertas",
        color="severidad",
        title="Productos con feedback registrado",
    )

    st.plotly_chart(fig, width="stretch")