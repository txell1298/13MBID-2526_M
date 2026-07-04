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
    add_expectation(
        "rango_edad",  # Verificar que la edad esté entre 18 y 100 años
        df_creditos["edad"].between(18, 100).all(),  # La validación a realizar
        "La edad debe estar entre 18 y 100 años."  # Mensaje de error en caso de que la validación falle
    )

    add_expectation(
        "situacion_vivienda",  # Verificar que la situación de vivienda sea una de las categorías válidas
        df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "HIPOTECA", "OTROS"]).all(),
        "La situación de vivienda no se encuentra en el rango válido."
    )
    
    # Atributo a analizar: Exactitud en datos de tarjetas
    add_expectation(
        "limite_credito_tc",
        df_tarjetas["limite_credito_tc"].between(0, 100000).all(),
        "El límite de crédito de la tarjeta debe estar entre 0 y 100000."
    )

    add_expectation(
        "personas_a_cargo",
        df_tarjetas["personas_a_cargo"].between(0, 10).all(),
        "El número de personas a cargo debe estar entre 0 y 10."
    )