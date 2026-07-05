# Se importan las librerías necesarias y se suprimen las advertencias
import pandas as pd
import numpy as np

def process_data(datos_creditos: str = "data/raw/datos_creditos.csv",
                 datos_tarjetas: str = "data/raw/datos_tarjetas.csv",
                 output_dir: str = "docs/processed/") -> None:
    """
    Lee los datos de créditos y tarjetas, realiza el procesamiendo

    Args:
        datos_creditos (str, optional): _description_. Defaults to "data/raw/datos_creditos.csv".
        datos_tarjetas (str, optional): _description_. Defaults to "data/raw/datos_tarjetas.csv".
        output_dir (str, optional): _description_. Defaults to "docs/processed/".
    """
    df_creditos = pd.read_csv(datos_creditos, sep=";")
    df_tarjetas = pd.read_csv(datos_tarjetas, sep=";")
    
    # Se filtran los datos para eliminar registros con edades superiores a 90 años
    df_creditos_filtrado = df_creditos.copy()
    df_creditos_filtrado = df_creditos_filtrado[df_creditos_filtrado['edad'] < 90]
    
    # Tratamiento de valores nulos para tasa interes
    df_creditos_filtrado['tasa_interes'] = df_creditos_filtrado.groupby("objetivo_credito")["tasa_interes"]\
        .transform(lambda x: x.fillna(x.median()))

    # Tratamiento de nulos para antiguedad_empleado
    df_creditos_filtrado['antiguedad_empleado'] = df_creditos_filtrado.groupby("edad")["antiguedad_empleado"]\
        .transform(lambda x: x.fillna(x.median()))
    
    # Se integran los datos de créditos y tarjetas utilizando el id_cliente como clave de unión
    df_integrado = pd.merge(df_creditos_filtrado, df_tarjetas, on='id_cliente', how='inner')

    # Capacidad de pago del cliente
    df_integrado["capacidad_pago"] = df_integrado["importe_solicitado"] / df_integrado["ingresos"]

    # El número de operaciones mensuales del cliente
    df_integrado["operaciones_mensuales"] = df_integrado["operaciones_ult_12m"] / 12

    # Presión financiera del cliente (mensual)
    df_integrado["presion_financiera"] = (
    (df_integrado["gastos_ult_12m"]/12 + df_integrado["importe_solicitado"]/(df_integrado["duracion_credito"] * 12))/ (df_integrado["ingresos"]/12))

    # Gasto promedio por operación realizada
    df_integrado["gasto_promedio_operacion"] = (
    df_integrado["gastos_ult_12m"] / df_integrado["operaciones_ult_12m"]
    )

    # Cantidad de operaciones mensuales con tarjeta
    df_integrado["operaciones_mensuales_tarjeta"] = (
    df_integrado["operaciones_ult_12m"] / 12
    )

    # Estabilidad laboral del cliente
    df_integrado["estabilidad_laboral"] = (
    df_integrado["antiguedad_empleado"] / df_integrado["edad"]
    )
    
    # Se eliminan las columnas originales y se integran las nuevas columnas procesadas
    columnas_a_eliminar = [
        "id_cliente",
        "operaciones_ult_12m",
        "importe_solicitado",
        "duracion_credito",
        "nivel_tarjeta",
    ]
    df_integrado.drop(columnas_a_eliminar, inplace=True, axis=1)
    
    
    # Exportar el DataFrame limpio a un nuevo archivo CSV
    import os

    os.makedirs("data/processed", exist_ok=True)
    df_integrado.to_csv("data/processed/datos_integrados.csv", index=False, sep=";")

    print("Dataset integrado guardado correctamente en data/processed/datos_integrados.csv")


if __name__ == "__main__":
    process_data() 