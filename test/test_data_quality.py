import pandas as pd
from pandera.pandas import DataFrameSchema, Column, Check
import pytest

@pytest.fixture
def datos_creditos():
    """Funcion para cargar los datos de créditos desde un archivo CSV.
    Returns:
        pd.DataFrame: DataFrame con los datos de créditos.
    """
    df = pd.read_csv("data/raw/datos_creditos.csv", sep=";")
    return df

@pytest.fixture
def datos_tarjetas():
    """Funcion para cargar los datos de tarjetas desde un archivo CSV.
    Returns:
        pd.DataFrame: DataFrame con los datos de tarjetas.
    """
    df = pd.read_csv("data/raw/datos_tarjetas.csv", sep=";")
    return df

def test_esquema_datos_creditos(datos_creditos):
    """Prueba para validar el esquema de los datos de créditos.
    Args:
        datos_creditos (pd.DataFrame): DataFrame con los datos de créditos.
    """
    esquema = DataFrameSchema({
        "id_cliente": Column(float, nullable=False),
        "edad": Column(int, Check.greater_than_or_equal_to(18), nullable=False),
        "importe_solicitado": Column(int, Check.greater_than(0), nullable=False),
        "duracion_credito": Column(int, Check.greater_than(0), nullable=False),
        "antiguedad_empleado": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "situacion_vivienda": Column(str, Check.isin(["ALQUILER", "PROPIA", "HIPOTECA", "OTROS"]), nullable=False),
        "ingresos": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "objetivo_credito": Column(str, Check.isin(["PERSONAL", "EDUCACIÓN", "SALUD", "INVERSIONES", "MEJORAS_HOGAR", "PAGO_DEUDAS" ]), nullable=False),        
        "pct_ingreso": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "tasa_interes": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "estado_credito": Column(int, Check.isin([0, 1]), nullable=False),
        "falta_pago": Column(str, Check.isin(["Y", "N"]), nullable=False),
    })

    esquema.validate(datos_creditos)


def test_esquema_datos_tarjetas(datos_tarjetas):
    """ Prueba para validar el esquema de los datos de tarjetas.
    Args:
        datos_tarjetas (pd.DataFrame): DataFrame con los datos de tarjetas.
    """
    esquema = DataFrameSchema({
        "id_cliente": Column(float, nullable=False),
        "antiguedad_cliente": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "estado_civil": Column(str, Check.isin(["CASADO", "SOLTERO", "DIVORCIADO", "DESCONOCIDO"]), nullable=False),
        "estado_cliente": Column(str, Check.isin(["ACTIVO", "PASIVO"]), nullable=False),
        "gastos_ult_12m": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "genero": Column(str, Check.isin(["M", "F"]), nullable=False),
        "limite_credito_tc": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "nivel_educativo": Column(str, nullable=False),
        "nivel_tarjeta": Column(str, Check.isin(["Blue", "Silver", "Gold", "Platinum"]), nullable=False),
        "operaciones_ult_12m": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "personas_a_cargo": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
    })

    esquema.validate(datos_tarjetas)

def test_basicos_creditos(datos_creditos):
    """
    Prueba para validar aspectos básicos de los datos de créditos.
    """
    df = datos_creditos

    # Exactitud: estructura esperada del dataset
    assert not df.empty, "El dataset de créditos está vacío."
    assert df.shape[1] == 12, (
        f"El dataset de créditos debería tener 12 columnas, pero tiene {df.shape[1]}."
    )

    # Completitud: se permite un máximo del 10% de valores nulos
    porcentaje_nulos = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    assert porcentaje_nulos <= 0.10, (
        f"El porcentaje de valores nulos en créditos supera la tolerancia del 10%: {porcentaje_nulos:.2%}"
    )

    # Exactitud: reglas básicas de negocio
    assert (df["edad"] >= 18).all(), "Existen clientes menores de 18 años."
    assert (df["importe_solicitado"] > 0).all(), "Existen importes solicitados no válidos."
    assert (df["duracion_credito"] > 0).all(), "Existen duraciones de crédito no válidas."
    assert (df["ingresos"] >= 0).all(), "Existen ingresos negativos."
    assert (df["pct_ingreso"] >= 0).all(), "Existen porcentajes de ingreso negativos."


def test_basicos_tarjetas(datos_tarjetas):
    """
    Prueba para validar aspectos básicos de los datos de tarjetas.
    """
    df = datos_tarjetas

    # Exactitud: estructura esperada del dataset
    assert not df.empty, "El dataset de tarjetas está vacío."
    assert df.shape[1] == 11, (
        f"El dataset de tarjetas debería tener 11 columnas, pero tiene {df.shape[1]}."
    )

    # Completitud: no se esperan valores nulos en tarjetas
    assert df.isnull().sum().sum() == 0, (
        "Existen valores nulos en el dataset de tarjetas."
    )

    # Exactitud: reglas básicas de negocio
    assert (df["antiguedad_cliente"] >= 0).all(), "Existen antigüedades de cliente negativas."
    assert (df["gastos_ult_12m"] >= 0).all(), "Existen gastos negativos."
    assert (df["limite_credito_tc"] >= 0).all(), "Existen límites de crédito negativos."
    assert (df["operaciones_ult_12m"] >= 0).all(), "Existen operaciones negativas."
    assert (df["personas_a_cargo"] >= 0).all(), "Existen valores negativos en personas a cargo."


def test_unicidad_id_cliente(datos_creditos, datos_tarjetas):
    """ Prueba para validar la unicidad del atributo clave id_cliente en ambos datasets.
    Args:
        datos_creditos (pd.DataFrame): DataFrame con los datos de créditos.
        datos_tarjetas (pd.DataFrame): DataFrame con los datos de tarjetas.
    """

    # Atributo a analizar: Consistencia
    # Validación: unicidad del identificador de cliente en ambos datasets

    assert datos_creditos["id_cliente"].is_unique, (
        "Existen valores duplicados de id_cliente en el dataset de créditos."
    )

    assert datos_tarjetas["id_cliente"].is_unique, (
        "Existen valores duplicados de id_cliente en el dataset de tarjetas."
    )


def test_integridad_referencial(datos_creditos, datos_tarjetas):
    """Prueba para validar la integridad referencial entre los datasets de créditos y tarjetas.
    Args:
        datos_creditos (pd.DataFrame): DataFrame con los datos de créditos.
        datos_tarjetas (pd.DataFrame): DataFrame con los datos de tarjetas.
    """

    df_ids = datos_creditos[["id_cliente"]].merge(
        datos_tarjetas[["id_cliente"]],
        on="id_cliente",
        how="outer",
        indicator=True
    )

    integridad_referencial = DataFrameSchema({
        "_merge": Column(
            str,
            Check.isin(["both"]),
            nullable=False
        )
    })

    integridad_referencial.validate(df_ids)
    