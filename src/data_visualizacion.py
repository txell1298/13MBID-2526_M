# Importación de librerías y supresión de advertencias
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def visualize_data(
    datos_creditos: str = "data/raw/datos_creditos.csv",
    datos_tarjetas: str = "data/raw/datos_tarjetas.csv",
    output_dir: str = "docs/figures",
) -> None:
    """
    Genera visualizaciones de los datos del escenario mediante gráficos de Seaborn y Matplotlib.

    Args:
        datos_creditos: Ruta al archivo CSV que contiene los datos de créditos.
        datos_tarjetas: Ruta al archivo CSV que contiene los datos de tarjetas.
        output_dir: Directorio donde se guardarán las figuras generadas.

    Returns:
        None
    """

    # Crear el directorio de salida si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Lectura de los datos
    df_creditos = pd.read_csv(datos_creditos, sep=";")
    df_tarjetas = pd.read_csv(datos_tarjetas, sep=";")

    sns.set_theme(style="whitegrid")

    # Visualización de la variable objetivo
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_creditos, x="falta_pago", palette="viridis")
    plt.title("Distribución de la variable objetivo")
    plt.xlabel("¿Presentó mora el cliente?")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    plt.savefig(output_path / "distribucion_target.png")
    plt.close()

    # Heatmap de correlaciones entre variables numéricas de créditos
    num_df = df_creditos.select_dtypes(include=["float64", "int64"])
    corr = num_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlaciones - Créditos")
    plt.tight_layout()
    plt.savefig(output_path / "correlacion_heatmap_creditos.png")
    plt.close()

    print(f"Imágenes guardadas correctamente en: {output_path.resolve()}")


if __name__ == "__main__":
    visualize_data()