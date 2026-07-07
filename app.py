import streamlit as st
import pandas as pd
from io import BytesIO
import joblib

st.title("Planner XGBoost")

archivo = st.file_uploader("Sube tu archivo excel",type=["xlsx","xls"])


import pandas as pd
import numpy as np
import streamlit as st


def predecir_por_tareas(
    df_proyecto_base,
    tareas_dict,
    modelo_cargado,
    df_modelos,
    df_patrones
):

    df_proyecto_base = df_proyecto_base.copy()

    if "ACRÓNIMO" not in df_proyecto_base.columns:
        st.error("El Excel debe contener la columna 'ACRÓNIMO'")
        return pd.DataFrame()

    filas = []

    # Generar una fila por cada tarea
    for tarea, cat_boq in tareas_dict.items():
        df_temp = df_proyecto_base.copy()
        df_temp["PLANIFICACIÓN"] = tarea
        df_temp["cat_boq"] = cat_boq
        filas.append(df_temp)

    df_input = pd.concat(filas, ignore_index=True)

    # ==========================
    # Predicción XGBoost
    # ==========================

    df_input["horas_xgb"] = modelo_cargado.predict(df_input)

    # ==========================
    # Reglas de negocio XGBoost
    # ==========================

    condicion_1 = (
        (df_input["FAMILIA"] == "SKID")
        & (df_input["cat_boq"] == "BCF Service - Panelling")
    )

    df_input.loc[condicion_1, "horas_xgb"] = 0

    trabajos_mt = [
        "TRANSFORMADOR",
        "TRABAJOS DE MEDIA TENSION",
        "TRABAJOS DE BT EN MT",
        "PRUEBAS DE MEDIA TENSION"
    ]

    condicion_2 = (
        (df_input["TRANSFORMER"] == "NO")
        & (df_input["PLANIFICACIÓN"].isin(trabajos_mt))
    )

    df_input.loc[condicion_2, "horas_xgb"] = 0

    # Evitar valores negativos
    df_input["horas_xgb"] = df_input["horas_xgb"].clip(lower=0)

    # ==========================
    # Selección de modelo
    # ==========================

    df_input = df_input.merge(
        df_modelos[
            ["FAMILIA", "TAMAÑO", "MODELO_ELEGIDO"]
        ],
        on=["FAMILIA", "TAMAÑO"],
        how="left"
    )

    # ==========================
    # Horas patrón
    # ==========================

    df_input = df_input.merge(
        df_patrones[
            ["FAMILIA", "TAMAÑO", "horas_patron"]
        ],
        on=["FAMILIA", "TAMAÑO"],
        how="left"
    )

    # Si no existe decisión, usar XGBoost por defecto
    df_input["MODELO_ELEGIDO"] = (
        df_input["MODELO_ELEGIDO"]
        .fillna("xgboost")
    )

    # ==========================
    # Horas finales
    # ==========================

    df_input["horas_predichas"] = np.where(
        df_input["MODELO_ELEGIDO"] == "patron",
        df_input["horas_patron"],
        df_input["horas_xgb"]
    )

    df_input["horas_predichas"] = (
        df_input["horas_predichas"]
        .fillna(0)
        .clip(lower=0)
    )

    # ==========================
    # Resultado
    # ==========================

    df_resultado = df_input[
        [
            "ACRÓNIMO",
            "FAMILIA",
            "TAMAÑO",
            "PLANIFICACIÓN",
            "cat_boq",
            "MODELO_ELEGIDO",
            "horas_predichas"
        ]
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


@st.cache_data
def cargar_df_modelos():
    return pd.read_csv("df_modelos.csv")


@st.cache_data
def cargar_df_patrones():
    return pd.read_csv("df_patrones.csv")


modelo = cargar_modelo()
df_modelos = cargar_df_modelos
df_patrones = cargar_df_patrones


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


if archivo:
    df = pd.read_excel(archivo)

    st.subheader("Datos cargados")
    st.dataframe(df)

    df_base = df.copy()

    if st.button("Predecir"):

        resultado = predecir_por_tareas(df_base, tareas_dict, modelo, df_modelos, df_patrones)

        st.subheader("Resultado detallado")
        st.dataframe(resultado)

        # ✅ Excel descargable
        excel_file = convertir_a_excel(resultado)

        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=excel_file,
            file_name="prediccion_horas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
