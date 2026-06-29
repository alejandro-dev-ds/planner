import streamlit as st
import pandas as pd
from io import BytesIO
import joblib

st.title("Planner XGBoost")

archivo = st.file_uploader("Sube tu archivo excel",type=["xlsx","xls"])


def predecir_por_tareas(df_proyecto_base, tareas_dict, modelo_cargado):

    df_proyecto_base = df_proyecto_base.copy()


    if "ACRÓNIMO" not in df_proyecto_base.columns:
        st.error("El Excel debe contener la columna 'ACRÓNIMO'")
        return pd.DataFrame()

    filas = []

    for tarea, cat_boq in tareas_dict.items():
        df_temp = df_proyecto_base.copy()
        df_temp["PLANIFICACIÓN"] = tarea
        df_temp["cat_boq"] = cat_boq
        filas.append(df_temp)

    df_input = pd.concat(filas, ignore_index=True)

    # Predicción
    horas = modelo_cargado.predict(df_input)
    df_input["horas_predichas"] = horas

    # Regla negocio
    condicion = (
        (df_input["FAMILIA"] == "SKID") &
        (df_input["cat_boq"] == "BCF Service - Panelling")
    )
    df_input.loc[condicion, "horas_predichas"] = 0

    # Evitar negativos
    df_input["horas_predichas"] = df_input["horas_predichas"].clip(lower=0)

    df_resultado = df_input[
        ["ACRÓNIMO", "PLANIFICACIÓN", "cat_boq", "horas_predichas"]
    ].copy()

    df_resultado = df_resultado.sort_values(
        ["ACRÓNIMO", "horas_predichas"],
        ascending=[True, False]
    )

    return df_resultado
    
def convertir_a_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_xgboost_planificacion_horas.pkl")

modelo = cargar_modelo()


tareas_dict = {

    # ELECTRICAL
    "CIERRE DE CUADROS Y PENDIENTES ELECTRICOS": "BCF Service - Electrical",
    "CIERRE DE CUADROS Y PENDIENTES MECANICOS": "BCF Service - Electrical",
    "CONEXIONADO DE EQUIPOS": "BCF Service - Electrical",
    "CONTROL DE ACCESO": "BCF Service - Electrical",
    "CORTE Y/O PREPARACION DE CABLE": "BCF Service - Electrical",
    "DESENSAMBLAJE INSTALACION ELECTRICA": "BCF Service - Electrical",
    "DESENSAMBLAJE INSTALACION MECANICA": "BCF Service - Electrical",
    "ENSAMBLAJE DE CUADROS CUADRISTA": "BCF Service - Electrical",
    "ETIQUETADO DE CABLE Y EQUIPOS": "BCF Service - Electrical",
    "INSTALACION DE CABLE": "BCF Service - Electrical",
    "LIMPIEZA FINAL CUADROS": "BCF Service - Electrical",
    "MECANIZADO DE ELEMENTOS DE INSTALACION ELECTRICA": "BCF Service - Electrical",
    "PRUEBAS ELECTRICAS CON TENSION": "BCF Service - Electrical",
    "PRUEBAS ELECTRICAS SIN TENSION": "BCF Service - Electrical",
    "SISTEMA DE ILUMINACION/EMERGENCIAS": "BCF Service - Electrical",
    "SISTEMA DE TIERRAS": "BCF Service - Electrical",
    "SOPORTE A COMMISSIONING": "BCF Service - Electrical",
    "TORQUEO": "BCF Service - Electrical",
    "TRABAJOS DE BT EN MT": "BCF Service - Electrical",
    "TRABAJOS DE MEDIA TENSION": "BCF Service - Electrical",
    "TRANSFORMADOR": "BCF Service - Electrical",

    # ASSEMBLY
    "CADENAS PORTACABLES": "BCF Service - Assembly",
    "CORTE Y ENSAMBLAJE DE CARRILES Y BANDEJAS": "BCF Service - Assembly",
    "ENSAMBLAJE DE EQUIPOS (CUADRO PRINCIPAL Y UPS)": "BCF Service - Assembly",
    "INSTALACION CARRILES DE SOPORTACION": "BCF Service - Assembly",
    "INSTALACION DE BANDEJAS": "BCF Service - Assembly",
    "INSTALACION DE ELEMENTOS Y/O EQUIPOS": "BCF Service - Assembly",
    "INSTALACION DE SUELO": "BCF Service - Assembly",
    "INSTALACION ROXTEC": "BCF Service - Assembly",
    "INTRODUCCION/FIJACION DE EQUIPOS": "BCF Service - Assembly",
    "MECANIZADO DE ELEMENTOS DE INSTALACION MECANICA": "BCF Service - Assembly",
    "NIVELACION": "BCF Service - Assembly",
    "TRABAJOS PREVIOS MECANICOS": "BCF Service - Assembly",

    # MONITORING
    "INSTALACION DE SENALES": "BCF Service - Monitoring",
    "INSTALACION DE UTP": "BCF Service - Monitoring",
    "MONTAJE DE EQUIPOS DE MONITORIZACION": "BCF Service - Monitoring",

    # COOLING
    "MONTAJE DE TUBERIAS": "BCF Service - Cooling",
    "REALIZACION DE PRUEBAS COOLING": "BCF Service - Cooling",
    "TRABAJOS DE CONEXIONADO Y VALVULERIA": "BCF Service - Cooling",
    "TRABAJOS DE FORRADO DE TUBERIAS": "BCF Service - Cooling",

    # FIRE SUPPRESSION
    "REALIZACION DE PRUEBAS PCI": "BCF Service - Fire Supression",
    "TRABAJOS DE INSTALACION PCI": "BCF Service - Fire Supression",

    # PACKING
    "EMBALAJE": "BCF Service - Packing",
    "LIMPIEZA FINAL": "BCF Service - Packing",

    # C/H CORRIDOR
    "CERRAMIENTO DE ALUMINIO": "BCF Service - C/H Corridor",

    # DOORS
    "INSTALACION DE PUERTAS": "BCF Service - Door/s",

    # PAINTING
    "REPASOS DE PINTURA": "BCF Service - Painting",

    # STRUCTURAL
    "PERFILERIA EXTERIOR": "BCF Service - Structural",

    # PANELLING
    "PANELADO": "BCF Service - Panelling",

    # MANTENIMIENTO
    "MANTENIMIENTO": "MANTENIMIENTO"
}



col1, col2 = st.columns([1, 2])

with col1:
    st.header("📂 Input")

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])

    ejecutar = False

    if archivo:
        if st.button("🚀 Ejecutar predicción", use_container_width=True):
            ejecutar = True

with col2:
    st.header("📊 Resultados")

# -------------------------
# EJECUCIÓN
# -------------------------

if archivo and ejecutar:

    df = pd.read_excel(archivo)

    with st.spinner("Calculando predicciones..."):
        resultado = predecir_por_tareas(df, tareas_dict, modelo)

    st.success("✅ Predicción completada")

    # KPIs
    total_horas = resultado["horas_predichas"].sum()
    num_proyectos = resultado["ACRÓNIMO"].nunique()

    m1, m2 = st.columns(2)
    m1.metric("⏱️ Total horas", f"{total_horas:,.0f}")
    m2.metric("📦 Nº proyectos", num_proyectos)

    # Filtro opcional
    proyectos = st.multiselect(
        "Filtrar por proyecto",
        options=resultado["ACRÓNIMO"].unique()
    )

    if proyectos:
        resultado = resultado[resultado["ACRÓNIMO"].isin(proyectos)]

    # Gráfico
    st.subheader("📈 Horas por tarea")

    grafico = resultado.groupby("PLANIFICACIÓN")["horas_predichas"].sum()
    st.bar_chart(grafico)

    # Tabla
    st.subheader("📋 Detalle")
    st.dataframe(resultado, use_container_width=True, hide_index=True)

    # Excel descarga
    excel_file = convertir_a_excel(resultado)

    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=excel_file,
        file_name="prediccion_horas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
