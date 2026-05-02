import streamlit as st

from views.general import mostrar_vista_general
from views.planeacion import mostrar_vista_planeacion
from views.finanzas import mostrar_vista_finanzas
from views.bi import mostrar_vista_bi
from views.operativa import mostrar_vista_operativa

from data_loader import cargar_datos_app

st.set_page_config(
    page_title="1C Company Retail Sales",
    page_icon="📈",
    layout="wide",
)

st.title("📈 1C Company")
st.caption("App de análisis, predicción y seguimiento de ventas retail by 1C Company")

df = cargar_datos_app()

tab_general, tab_planeacion, tab_finanzas, tab_bi, tab_operativa = st.tabs(
    [
        "Análisis general",
        "Dirección de planeación",
        "Finanzas",
        "BI",
        "Operativa",
    ]
)

with tab_general:
    mostrar_vista_general(df)

with tab_planeacion:
    mostrar_vista_planeacion(df)

with tab_finanzas:
    mostrar_vista_finanzas(df)

with tab_bi:
    mostrar_vista_bi(df)

with tab_operativa:
    mostrar_vista_operativa(df)
