import streamlit as st
import pandas as pd
from io import BytesIO
import joblib
import numpy as np
import plotly.express as px

st.title("Planner XGBoost")

archivo = st.file_uploader("Sube tu archivo excel",type=["xlsx","xls"])


def predecir_por_tareas(
    df_proyecto_base,
    tareas_dict,
    fases,
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

    excepciones_skid = [
        "BCF Service - Panelling",
        "BCF Service - Fire Supression",
        "BCF Service - Cooling",
    ]

    condicion_1 = (
        (df_input["FAMILIA"] == "SKID")
        & (df_input["cat_boq"].isin(excepciones_skid))
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
            ["FAMILIA", "TAMAÑO", "PLANIFICACIÓN", "MODELO_ELEGIDO"]
        ],
        on=["FAMILIA", "TAMAÑO", "PLANIFICACIÓN"],
        how="left"
    )

    # ==========================
    # Horas patrón
    # ==========================

    df_input = df_input.merge(
        df_patrones[
            ["FAMILIA", "TAMAÑO", "PLANIFICACIÓN", "Horas_Patron"]
        ],
        on=["FAMILIA", "TAMAÑO", "PLANIFICACIÓN"],
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
        df_input["Horas_Patron"],
        df_input["horas_xgb"]
    )

    df_input["horas_predichas"] = (
        df_input["horas_predichas"]
        .fillna(0)
        .clip(lower=0)
    )

    # ==========================
    # Tareas Quality fijas
    # ==========================

    tareas_qc = {
        "1981-ForQ-en-002 QC incoming inspection of NON-ISO modules": 1.5,
        "1981-ForQ-en-005 Incoming inspection of Base Design": 2,
        "1981-ForQ-en-006 QC Pre-instalation of doors": 0.25,
        "1981-ForQ-en-003 Incoming inspection of electrical panels": 1,
        "1981-ForQ-en-001 Incoming Inspection of HV Cells": 0.5,
        "1981-ForQ-en-004 Incoming-inspection-HV-MV-Transformer": 0.5,
        "1981-ForQ-en-007 Acceptance of Installations. Assembly": 1.5,
        "1981-ForQ-en-008 Acceptance of installations MDC. Roof Sealing": 0.5,
        "1981-ForQ-en-011 Acceptance of Installations. FSS": 1,
        "1981-ForQ-en-013 Acceptance of Installations. Cooling": 1,
        "1981-ForQ-en-010 Acceptance of Installations. Electrical": 2.5,
        "1981-ForQ-en-012 Acceptance of torque": 2,
        "1981-ForQ-en-016 Acceptance of installations. Monitoring": 0.75,
        "1981-ForQ-en-015 Door Fan Test": 0.75,
        "1981-ForQ-en-018 QC 100% MDC": 1,
        "1981-ForQ-en-028 Surface damage": 0.5,
        "1981-ForQ-en-017 Outgoing electrical panel visual inspection": 0.75,
        "1981-ForQ-en-020 Pre Shipment inspection MDC": 2,
        "1981-ForQ-en-019 Authorization for the release of the solutions": 0.5,
    }

    filas_qc = []

    for tarea, horas in tareas_qc.items():
        df_temp = df_proyecto_base.copy()
        df_temp["PLANIFICACIÓN"] = tarea
        df_temp["cat_boq"] = "QC"
        df_temp["MODELO_ELEGIDO"] = "fijo"
        df_temp["horas_predichas"] = horas
        filas_qc.append(df_temp)

    df_qc = pd.concat(filas_qc, ignore_index=True)

    df_input = pd.concat(
        [df_input, df_qc],
        ignore_index=True,
        sort=False
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

    df_resultado["FASE"] = (
        df_resultado["PLANIFICACIÓN"]
        .map(fases)
        .fillna(999)
    )

    df_resultado = df_resultado.sort_values(
        ["ACRÓNIMO", "FASE"],
        ascending=[True, True]
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
    df = pd.read_csv("df_modelos.csv", sep=";")
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def cargar_df_patrones():
    df = pd.read_csv(
        "df_patrones.csv",
        sep=";",
        encoding="cp1252"
    )

    st.write(df.columns.tolist())

    return df

modelo = cargar_modelo()
df_modelos = cargar_df_modelos()
df_patrones = cargar_df_patrones()


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

fases = {
    "TRABAJOS PREVIOS MECANICOS": 100,
    "CORTE Y ENSAMBLAJE DE CARRILES Y BANDEJAS": 105,
    "NIVELACION": 110,
    "INSTALACION CARRILES DE SOPORTACION": 115,
    "INSTALACION DE BANDEJAS": 120,
    "MECANIZADO DE ELEMENTOS DE INSTALACION MECANICA": 125,
    "EQUIPOS": 130,
    "INTRODUCCION/FIJACION DE EQUIPOS": 131,
    "ENSAMBLAJE DE CUADROS CUADRISTA": 132,
    "ENSAMBLAJE DE EQUIPOS (CUADRO PRINCIPAL Y UPS)": 133,
    "CADENAS PORTACABLES": 134,
    "INSTALACION DE SUELO": 135,
    "PANELADO": 140,
    "MODIFICACIONES EN PANEL": 145,
    "R-QP05-12 Pre-Installation of doors": 146,
    "INSTALACION DE PUERTAS": 147,
    "INSTALACION ROXTEC": 148,
    "INSTALACION DE ELEMENTOS Y/O EQUIPOS": 149,
    "CERRAMIENTO DE ALUMINIO": 150,
    "PERFILERIA EXTERIOR": 155,
    "R-QP05-28 Prefabricated Module Roof Inspection": 160,
    "CORTE Y/O PREPARACION DE CABLE": 165,
    "MECANIZADO DE ELEMENTOS DE INSTALACION ELECTRICA": 170,
    "SISTEMA DE TIERRAS": 175,
    "SISTEMA DE ILUMINACION/EMERGENCIAS": 180,
    "CABLEADO Y CONEXIONADO POTENCIA": 185,
    "INSTALACION DE CABLE": 186,
    "CONEXIONADO DE EQUIPOS": 187,
    "ETIQUETADO DE CABLE Y EQUIPOS": 188,
    "INSTALACION DE MEDIA TENSION": 190,
    "TRANSFORMADOR": 191,
    "TRABAJOS DE MEDIA TENSION": 192,
    "TRABAJOS DE BT EN MT": 193,
    "PRUEBAS DE MEDIA TENSION": 194,
    "SISTEMA DE MONITORIZACION": 195,
    "MONTAJE DE EQUIPOS DE MONITORIZACION": 196,
    "INSTALACION DE SENALES": 197,
    "INSTALACION DE UTP": 198,
    "R-QP05-16 Acceptance of Installations. Monitoring": 199,
    "CONTROL DE ACCESO": 200,
    "PRUEBAS ELECTRICAS PRODUCCION": 205,
    "RECOLECCIÓN DE NUMEROS DE SERIE": 206,
    "PRUEBAS ELECTRICAS SIN TENSION": 207,
    "TORQUEO": 208,
    "R-QP05-11 Acceptance of Installations. Electrical": 209,
    "R-QP05-40 Acceptance of Torque": 210,
    "R-QP07-08 BCF Factory Power Energization Authorization": 211,
    "PRUEBAS ELECTRICAS CON TENSION": 212,
    "SOPORTE A COMMISSIONING": 213,
    "TRABAJOS DE INSTALACION PCI": 215,
    "REALIZACION DE PRUEBAS PCI": 220,
    "R-QP05-27 Door Fan Test": 221,
    "R-QP05-04 Acceptance of Installations. FSS": 222,
    "R-QP05-18 Acceptance of Installations. Assembly": 223,
    "MONTAJE DE TUBERIAS": 225,
    "TRABAJOS DE CONEXIONADO Y VALVULERIA": 230,
    "TRABAJOS DE FORRADO DE TUBERIAS": 235,
    "REALIZACION DE PRUEBAS COOLING": 240,
    "R-QP05-05 Acceptance of Installations. Cooling": 241,
    "TRASLADO Y UBICACION EN ZONA CX": 245,
    "TESTEO Cx": 250,
    "UPS START UP": 255,
    "FAT/FOK/Witness test": 260,
    "TRASLADO Y UBICACION EN ZONA PRODUCCION.": 265,
    "MODIFICACIONES FAT/FOK/WITNESS TEST": 270,
    "R-QP05-03 QC 100% MDC": 271,
    "DESENSAMBLAJE INSTALACION ELECTRICA": 275,
    "DESENSAMBLAJE INSTALACION MECANICA": 280,
    "LIMPIEZA FINAL CUADROS": 285,
    "R-QP05-29 Outgoing electrical panel visual inspection": 286,
    "CIERRE DE CUADROS Y PENDIENTES ELECTRICOS": 290,
    "CIERRE DE CUADROS Y PENDIENTES MECANICOS": 295,
    "LIMPIEZA FINAL": 300,
    "REPASOS DE PINTURA": 305,
    "R-QP05-02 Pre Shipment Inspection MDC": 306,
    "R-QP05-24 Authorization for the release of the solutions": 307,
    "EMBALAJE": 310,
    "CARGA": 315,
    "MANTENIMIENTO": 9999
}


if archivo:
    df = pd.read_excel(archivo)

    st.subheader("Datos cargados")

    df_base = df.copy()

    # Calcular predicción una sola vez y guardarla
    if st.button("Predecir"):

        st.session_state["resultado"] = predecir_por_tareas(
            df_base,
            tareas_dict,
            fases,
            modelo,
            df_modelos,
            df_patrones
        )

    # Mostrar resultados si existen
    if "resultado" in st.session_state:

        resultado = st.session_state["resultado"]

        st.subheader("Visualización")

        proyectos = sorted(resultado["ACRÓNIMO"].unique())

        proyectos_seleccionados = st.multiselect(
            "Selecciona los proyectos a visualizar",
            proyectos,
            default=proyectos[:1]
        )

        if len(proyectos_seleccionados) > 0:

            df_graf = resultado[
                resultado["ACRÓNIMO"].isin(proyectos_seleccionados)
            ].copy()

            fig = px.bar(
                df_graf,
                x="PLANIFICACIÓN",
                y="horas_predichas",
                color="ACRÓNIMO",
                barmode="group",
                title="Horas previstas por tarea",
                labels={
                    "PLANIFICACIÓN": "Tarea",
                    "horas_predichas": "Horas"
                }
            )

            fig.update_layout(
                height=700,
                xaxis_tickangle=-45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Tabla resumen opcional
            resumen = (
                df_graf
                .pivot_table(
                    index="PLANIFICACIÓN",
                    columns="ACRÓNIMO",
                    values="horas_predichas",
                    aggfunc="sum"
                )
                .fillna(0)
            )

            st.subheader("Comparativa por tarea")
            st.dataframe(resumen)

        else:
            st.info("Seleccione al menos un proyecto para visualizar.")

        st.subheader("Resultado detallado")
        st.dataframe(resultado)

        # Excel descargable
        excel_file = convertir_a_excel(resultado)

        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=excel_file,
            file_name="prediccion_horas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
