import streamlit as st


def mostrar_vista_finanzas() -> None:
    st.header("Finanzas")
    st.write("Vista enfocada en impacto financiero, margen, presupuesto y variaciones.")

    st.info("Aquí puedes agregar ventas monetarias, desviaciones presupuestales y rentabilidad.")

#df = cargar_datos_dummy()
#df_filtrado = df[df["tienda"].isin(tiendas)]
#st.subheader("Datos base")
#st.dataframe(df_filtrado, use_container_width=True)