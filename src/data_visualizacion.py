import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

warnings.filterwarnings("ignore")


def visualize_data(
    datos_creditos: str = "data/raw/datos_creditos.csv",
    datos_tarjetas: str = "data/raw/datos_tarjetas.csv",
    output_dir: str = "docs/figures/"
) -> None:
    """
    Generar visualizaciones de los datos del escenario
    mediante gráficos de Seaborn, Matplotlib y reportes automáticos con ydata-profiling.

    Args:
        datos_creditos (str): Ruta al archivo CSV que contiene los datos de créditos.
        datos_tarjetas (str): Ruta al archivo CSV que contiene los datos de tarjetas.
        output_dir (str): Directorio donde se guardarán las figuras generadas.

    Returns:
        None
    """

    # Crear directorios de salida si no existen
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reports_path = Path("docs/reports")
    reports_path.mkdir(parents=True, exist_ok=True)

    # Lectura de los datos
    df_creditos = pd.read_csv(datos_creditos, sep=";")
    df_tarjetas = pd.read_csv(datos_tarjetas, sep=";")

    sns.set_theme(style="whitegrid")

    target_col = "falta_pago"

    # 1. Gráfico distribución de la variable target
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_creditos, x=target_col, palette="viridis")
    plt.title("Distribución de la variable objetivo")
    plt.xlabel("¿Presentó mora el cliente?")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    plt.savefig(output_path / "distribucion_target.png")
    plt.close()

    # 2. Heatmap de correlaciones entre variables numéricas de créditos
    num_df_creditos = df_creditos.select_dtypes(include=["float64", "int64"])
    corr_creditos = num_df_creditos.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_creditos, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlaciones - Créditos")
    plt.tight_layout()
    plt.savefig(output_path / "correlacion_heatmap_creditos.png")
    plt.close()

    # 3. Heatmap de correlaciones entre variables numéricas de tarjetas
    num_df_tarjetas = df_tarjetas.select_dtypes(include=["float64", "int64"])
    corr_tarjetas = num_df_tarjetas.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_tarjetas, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlaciones - Tarjetas")
    plt.tight_layout()
    plt.savefig(output_path / "correlacion_heatmap_tarjetas.png")
    plt.close()

    # 4. Boxplot de una variable numérica respecto al target
    numeric_cols = [
        col for col in df_creditos.select_dtypes(include=["float64", "int64"]).columns
        if col != target_col
    ]

    plt.figure(figsize=(10, 6))

    if numeric_cols:
        selected_num_col = numeric_cols[0]
        sns.boxplot(data=df_creditos, x=target_col, y=selected_num_col, palette="viridis")
        plt.title(f"{selected_num_col} según la variable objetivo")
        plt.xlabel("¿Presentó mora el cliente?")
        plt.ylabel(selected_num_col)
    else:
        plt.text(
            0.5,
            0.5,
            "No hay variables numéricas disponibles para generar el boxplot.",
            ha="center",
            va="center",
            fontsize=12
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path / "boxplot_variable_numerica_vs_target.png")
    plt.close()

    # 5. Heatmap de valores nulos del dataset de créditos
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_creditos.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Mapa de valores nulos - Dataset de créditos")
    plt.xlabel("Variables")
    plt.tight_layout()
    plt.savefig(output_path / "heatmap_valores_nulos_creditos.png")
    plt.close()

    # 6. Reporte automático con ydata-profiling - Créditos
    profile_creditos = ProfileReport(
        df_creditos,
        title="Reporte EDA - Datos de créditos",
        explorative=True
    )

    profile_creditos.to_file(reports_path / "reporte_eda_creditos.html")

    # 7. Reporte automático con ydata-profiling - Tarjetas
    profile_tarjetas = ProfileReport(
        df_tarjetas,
        title="Reporte EDA - Datos de tarjetas",
        explorative=True
    )

    profile_tarjetas.to_file(reports_path / "reporte_eda_tarjetas.html")

    print(f"Imágenes guardadas correctamente en: {output_path.resolve()}")
    print(f"Reportes guardados correctamente en: {reports_path.resolve()}")


if __name__ == "__main__":
    visualize_data()