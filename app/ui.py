import streamlit as st
import requests

st.set_page_config(
    page_title = "Predicción de Mora en Créditos",
    page_icon = ":credit_card:",
    layout="wide"
)

with st.sidebar:
    st.header("Instrucciones")
    st.write("""
    1. Ingrese los datos del cliente en el formulario.
    2. Haga clic en el botón "Predecir" para obtener la probabilidad de mora en créditos.
    3. Revise los resultados y la información del modelo.
    """)

    st.divider()
    st.header("Configuración de la API")
    api_url = st.text_input(
        "URL de la API",
        value="http://localhost:8000",
        help="Ingrese la URL donde está alojada la API de predicción."
    )

    st.divider()
    if st.button("Probar Conexión a la API"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("Conexión exitosa a la API.")
            else:
                st.error(f"Error al conectar con la API: {response.status_code}")
        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")


st.title("Predicción de Mora en Créditos")
st.write("Ingrese los datos del cliente para predecirla probabilidad de mora en créditos utilizando nuestro modelo de machine learning entrenado con datos històricos.")

 # Formulario de entrada de datos
# Formulario de entrada de datos
with st.form("prediction_form"):

    st.subheader("Datos del Cliente")
    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.number_input(
            "Edad",
            min_value=18,
            max_value=100,
            value=30
        )

        genero = st.selectbox(
            "Género",
            options=["M", "F"]
        )

        estado_civil = st.selectbox(
            "Estado Civil",
            options=["CASADO", "SOLTERO", "DIVORCIADO", "VIUDO"]
        )

    with col2:
        nivel_educativo = st.selectbox(
            "Nivel Educativo",
            options=[
                "PRIMARIO",
                "SECUNDARIO",
                "TERCIARIO",
                "UNIVERSITARIO_INCOMPLETO",
                "UNIVERSITARIO_COMPLETO"
            ]
        )

        situacion_vivienda = st.selectbox(
            "Situación de Vivienda",
            options=["ALQUILER", "PROPIETARIO", "HIPOTECADA", "OTRA"]
        )

        personas_a_cargo = st.number_input(
            "Personas a Cargo",
            min_value=0.0,
            max_value=20.0,
            value=0.0
        )

    with col3:
        estado_cliente = st.selectbox(
            "Estado del Cliente",
            options=["ACTIVO", "INACTIVO"]
        )

        estado_credito = st.number_input(
            "Estado del Crédito",
            min_value=0,
            max_value=1,
            value=1,
            step=1
        )

        falta_pago = st.selectbox(
            "Falta de pago",
            options=["N", "Y"]
        )

    st.divider()

    st.subheader("Información Financiera y Laboral")
    col4, col5, col6 = st.columns(3)

    with col4:
        ingresos = st.number_input(
            "Ingresos",
            min_value=0,
            value=50000
        )

        antiguedad_empleado = st.number_input(
            "Antigüedad del Empleado (años)",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        limite_credito_tc = st.number_input(
            "Límite crédito tarjeta",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )

    with col5:
        objetivo_credito = st.selectbox(
            "Objetivo del Crédito",
            options=["PERSONAL", "VIVIENDA", "VEHICULO", "NEGOCIOS", "EDUCACION", "OTRO"]
        )

        tasa_interes = st.number_input(
            "Tasa de interés (%)",
            min_value=0.0,
            value=15.0,
            step=0.1
        )

        pct_ingreso = st.number_input(
            "Porcentaje de Ingreso (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=0.1
        )

    with col6:
        capacidad_pago = st.number_input(
            "Capacidad de pago",
            min_value=0.0,
            value=0.50,
            step=0.01
        )

        presion_financiera = st.number_input(
            "Presión financiera",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        antiguedad_cliente = st.number_input(
            "Antigüedad cliente (meses)",
            min_value=0.0,
            value=12.0,
            step=1.0
        )

    st.divider()

    st.subheader("Actividad del Cliente")
    col7, col8, col9 = st.columns(3)

    with col7:
        gastos_ult_12m = st.number_input(
            "Gastos últimos 12 meses",
            min_value=0.0,
            value=1200.0,
            step=100.0
        )

    with col8:
        operaciones_mensuales = st.number_input(
            "Operaciones mensuales",
            min_value=0.0,
            value=3.5,
            step=0.5
        )

    with col9:
        estabilidad_laboral = st.number_input(
            "Estabilidad laboral",
            min_value=0.0,
            value=0.25,
            step=0.01
        )

    submitted = st.form_submit_button("Predecir")

# En este proyecto, el formulario recoge variables del cliente, información financiera,
# situación laboral y actividad bancaria, como edad, ingresos, tasa de interés,
# estado del crédito, gastos de los últimos 12 meses y operaciones mensuales.
#
# EXTRA: algunas variables derivadas no se piden directamente al usuario, sino que se calculan
#        antes de realizar la predicción. Por ejemplo:
#
#        - operaciones_mensuales_tarjeta se obtiene a partir de operaciones_mensuales.
#        - gasto_promedio_operacion se calcula dividiendo gastos_ult_12m entre las operaciones
#          estimadas del año.
#
#        Estas transformaciones se realizan en la API antes de llamar al modelo, para que
#        el usuario solo introduzca los datos originales y el sistema genere las variables
#        necesarias para la predicción.

if submitted:
    input_data = {
        "edad": edad,
        "antiguedad_empleado": antiguedad_empleado,
        "situacion_vivienda": situacion_vivienda,
        "ingresos": ingresos,
        "objetivo_credito": objetivo_credito,
        "pct_ingreso": pct_ingreso,
        "tasa_interes": tasa_interes,
        "estado_credito": estado_credito,
        "falta_pago": falta_pago,
        "antiguedad_cliente": antiguedad_cliente,
        "estado_civil": estado_civil,
        "estado_cliente": estado_cliente,
        "gastos_ult_12m": gastos_ult_12m,
        "genero": genero,
        "limite_credito_tc": limite_credito_tc,
        "nivel_educativo": nivel_educativo,
        "personas_a_cargo": personas_a_cargo,
        "capacidad_pago": capacidad_pago,
        "operaciones_mensuales": operaciones_mensuales,
        "presion_financiera": presion_financiera,
        "estabilidad_laboral": estabilidad_laboral
    }

    try:
        resp = requests.post(
            f"{api_url}/predict",
            json=input_data,
            timeout=10
        )

        resp.raise_for_status()
        result = resp.json()

        st.divider()
        st.subheader("📊 Resultado de la predicción")

        prediction = result["prediction"]
        prob = result.get("probability", {})
        labels = result.get(
            "class_labels",
            {
                "0": "No entra en mora",
                "1": "Entra en mora"
            }
        )

        label_text = labels.get(str(prediction), prediction)

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if str(prediction) == "1":
                st.error(f"**Predicción: {label_text}**")
            else:
                st.success(f"**Predicción: {label_text}**")

        with col_res2:
            prob_mora = prob.get("1", prob.get(str(prediction), 0))
            prob_no_mora = prob.get("0", 1 - prob_mora)

            st.metric(
                "Probabilidad de mora",
                f"{prob_mora * 100:.1f}%"
            )

            st.metric(
                "Probabilidad de no mora",
                f"{prob_no_mora * 100:.1f}%"
            )

        with st.expander("Ver respuesta completa de la API"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API. Verificá la URL en el panel lateral.")

    except requests.exceptions.HTTPError as e:
        st.error(
            f"Error de la API ({resp.status_code}): "
            f"{resp.json().get('detail', str(e))}"
        )

    except Exception as e:
        st.error(f"Error inesperado: {e}")