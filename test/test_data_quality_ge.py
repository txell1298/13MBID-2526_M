import pandas as pd
from pathlib import Path
import pytest
import warnings


warnings.filterwarnings(
    "ignore",
    message=r".*Number.* field should not be instantiated.*",
)

import great_expectations as ge


pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Number.*should not be instantiated.*"),
    pytest.mark.filterwarnings("ignore:.*result_format.*Validator-level.*"),
    pytest.mark.filterwarnings("ignore:.*result format.*Expectation-level.*"),
]


# Paths
PROJECT_DIR = Path(".").resolve()
DATA_DIR = PROJECT_DIR / "data"


def test_great_expectations():
    """
    Prueba para validar la calidad de los datos utilizando Great Expectations.
    """

    # Cargar los datos de créditos y tarjetas
    df_creditos = pd.read_csv(DATA_DIR / "raw/datos_creditos.csv", sep=";")
    df_tarjetas = pd.read_csv(DATA_DIR / "raw/datos_tarjetas.csv", sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }

    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1
        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })

    # Atributo a analizar: Exactitud (rangos de valores en datos)
    # Validación 1: RAngo de edad (18-100 años)
    edad_valida = df_creditos["edad"].between(18, 100).all()
    mensaje_edad = ""
    if not edad_valida:
        edades_fuera = df_creditos[(df_creditos["edad"]<18)| (df_creditos["edad"]>100)]["edad"].unique()
        mensaje_edad = f"Edades fuera de rango encontradas:{sorted(edades_fuera)}."
    add_expectation(
        "rango_edad",  # Verificar que la edad esté entre 18 y 100 años
        edad_valida,
        f"La edad debe estar entre 18 y 100 años. {mensaje_edad}"  # Mensaje de error en caso de que la validación falle
    )

    # Validación 2: Rango de valores para situación de vivienda (ALQUILER, PROPIA, OTROS, HIPOTECA)
    vivienda_valida = df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "OTROS", "HIPOTECA"]).all()
    mensaje_vivienda = ""

        # Validación 2: Rango de valores para situación de vivienda
    vivienda_valida = df_creditos["situacion_vivienda"].isin(
        ["ALQUILER", "PROPIA", "OTROS", "HIPOTECA"]
    ).all()

    mensaje_vivienda = ""

    if not vivienda_valida:
        viviendas_fuera = df_creditos[
            ~df_creditos["situacion_vivienda"].isin(
                ["ALQUILER", "PROPIA", "OTROS", "HIPOTECA"]
            )
        ]["situacion_vivienda"].unique()

        mensaje_vivienda = (
            f"Situaciones de vivienda no válidas encontradas: "
            f"{sorted(viviendas_fuera)}"
        )

    add_expectation(
        "situacion_vivienda",
        vivienda_valida,
        f"La situación de vivienda no se encuentra en el rango válido. {mensaje_vivienda}"
    )

    ##########################################################################
    # TODO: Agregar al menos dos (2) validaciones más para el dataset de tarjetas.
    ##########################################################################

    # Validación 3: Rango de valores para límite de crédito de tarjeta
    limite_credito_valido = df_tarjetas["limite_credito_tc"].between(0, 100000).all()
    mensaje_limite_credito = ""

    if not limite_credito_valido:
        limite_credito_fuera = df_tarjetas[
            ~df_tarjetas["limite_credito_tc"].between(0, 100000)
        ]["limite_credito_tc"]

        mensaje_limite_credito = (
            f"Existen {len(limite_credito_fuera)} registros con límite de crédito "
            f"fuera del rango 0-100000."
        )

    add_expectation(
        "limite_credito_tc",
        limite_credito_valido,
        f"El límite de crédito de la tarjeta debe estar entre 0 y 100000. {mensaje_limite_credito}"
    )

    # Validación 4: Rango de valores para personas a cargo
    personas_cargo_valido = df_tarjetas["personas_a_cargo"].between(0, 10).all()
    mensaje_personas_cargo = ""

    if not personas_cargo_valido:
        personas_cargo_fuera = df_tarjetas[
            ~df_tarjetas["personas_a_cargo"].between(0, 10)
        ]["personas_a_cargo"]

        mensaje_personas_cargo = (
            f"Existen {len(personas_cargo_fuera)} registros con personas a cargo "
            f"fuera del rango 0-10."
        )

    add_expectation(
        "personas_a_cargo",
        personas_cargo_valido,
        f"El número de personas a cargo debe estar entre 0 y 10. {mensaje_personas_cargo}"
    )