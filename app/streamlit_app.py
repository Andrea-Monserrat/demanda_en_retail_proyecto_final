import streamlit as st

from views.general import mostrar_vista_general
from views.planeacion import mostrar_vista_planeacion
from views.finanzas import mostrar_vista_finanzas
from views.bi import mostrar_vista_bi
from views.operativa import mostrar_vista_operativa




st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Retail Sales Forecasting")
st.caption("App de análisis, predicción y seguimiento de ventas retail")

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
    mostrar_vista_general()

with tab_planeacion:
    mostrar_vista_planeacion()

with tab_finanzas:
    mostrar_vista_finanzas()

with tab_bi:
    mostrar_vista_bi()

with tab_operativa:
    mostrar_vista_operativa()