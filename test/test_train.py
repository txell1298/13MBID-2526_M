from src.train_model import train_model
import json
from pathlib import Path
import pytest


def test_train_model(tmp_path):
    # Se establece la raíz del proyecto para acceder a los archivos del proyecto
    project_root = Path(__file__).resolve().parents[1]

    baseline_path = project_root / "docs" / "metrics" / "train_metrics.json"
    data_path = project_root / "docs" / "processed" / "datos_integrados.csv"

    if not baseline_path.exists():
        pytest.skip("Baseline metrics file not found. Run train_model.py to generate it.")

    assert data_path.exists(), f"No existe el archivo de datos: {data_path}"

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # Rutas temporales para que el test no sobrescriba los archivos reales del proyecto
    model_path = tmp_path / "prod_model.pkl"
    preprocessor_path = tmp_path / "prod_preprocessor.pkl"
    metrics_path = tmp_path / "train_metrics.json"

    # Ejecutar el entrenamiento
    _, _, metrics = train_model(
        data_path=str(data_path),
        model_path=str(model_path),
        preprocessor_path=str(preprocessor_path),
        metrics_path=str(metrics_path)
    )

    # Comprobar que se han generado los archivos esperados
    assert model_path.exists(), "No se ha generado el modelo."
    assert preprocessor_path.exists(), "No se ha generado el preprocesador."
    assert metrics_path.exists(), "No se ha generado el archivo de métricas."

    # Comprobar que las métricas tienen las mismas claves
    assert set(metrics.keys()) == set(baseline.keys()), (
        "Las métricas generadas no coinciden con las métricas de referencia."
    )

    # Comparar valores de métricas
    atol = 1e-9
    for k in baseline.keys():
        assert metrics[k] == pytest.approx(baseline[k], rel=0, abs=atol), (
            f"La métrica {k} cambió: baseline={baseline[k]}, actual={metrics[k]}"
        )
        
        
if __name__ == "__main__":
    test_train_model()