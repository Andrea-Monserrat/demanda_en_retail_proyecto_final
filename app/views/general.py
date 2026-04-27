import streamlit as st


def mostrar_vista_general() -> None:
    st.header("Análisis general")
    st.write("Vista ejecutiva general del desempeño de ventas.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Ventas totales", "$0")
    col2.metric("Forecast total", "$0")
    col3.metric("Desviación", "0%")

    st.info("Aquí puedes agregar KPIs generales, gráficas globales y resumen del modelo.")