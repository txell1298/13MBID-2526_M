# Se importan las librerías necesarias y se suprimen las advertencias
import json
import joblib
import warnings

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from mlflow.models.signature import infer_signature

from scipy.stats import randint
from scipy.stats import uniform

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(data_path: str = "docs/processed/datos_integrados.csv"):
    """
    Carga los datos desde un archivo CSV y devuelve los conjuntos de entrenamiento,
    validación y test.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos.

    Returns:
        Tupla con X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y.
    """

    df = pd.read_csv(data_path, sep=";")

    # Se divide el dataset en variables predictoras y variable objetivo
    target = "falta_pago"

    features_X = df.drop(columns=[target])
    labels_y = df[target]

    # Se genera el conjunto de entrenamiento, validación y test con estratificación
    # Primero se separa el conjunto de test final, equivalente al 10% del total
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X,
        labels_y,
        test_size=0.10,
        random_state=42,
        stratify=labels_y
    )

    # Después se separan entrenamiento y validación
    # El 22% del 90% equivale aproximadamente al 20% del total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.22,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y


def create_preprocessor(features_X: pd.DataFrame):
    """
    Crea un preprocesador para las variables numéricas y categóricas.

    Args:
        features_X (pd.DataFrame): DataFrame con las variables predictoras.

    Returns:
        ColumnTransformer: Preprocesador configurado.
    """

    # Identificar columnas numéricas y categóricas
    num_cols = features_X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = features_X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    return preprocessor


def encode_target(y_train, y_val, y_test):
    """
    Codifica la variable objetivo si contiene valores tipo N/Y.

    Args:
        y_train: Variable objetivo de entrenamiento.
        y_val: Variable objetivo de validación.
        y_test: Variable objetivo de test.

    Returns:
        y_train_eval, y_val_eval, y_test_eval codificadas.
    """

    if set(y_train.dropna().unique()) == {"N", "Y"}:
        y_train_eval = y_train.map({"N": 0, "Y": 1})
        y_val_eval = y_val.map({"N": 0, "Y": 1})
        y_test_eval = y_test.map({"N": 0, "Y": 1})
    else:
        y_train_eval = y_train.copy()
        y_val_eval = y_val.copy()
        y_test_eval = y_test.copy()

    return y_train_eval, y_val_eval, y_test_eval


def train_model(
    data_path: str = "docs/processed/datos_integrados.csv",
    model_path: str = "docs/models/best_model.pkl",
    preprocessor_path: str = "docs/models/prod_preprocessor.pkl",
    metrics_path: str = "docs/metrics/train_metrics.json"
):
    """
    Entrena un modelo para predecir la variable objetivo falta_pago.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos.
        model_path (str): Ruta donde se guardará el modelo entrenado.
        preprocessor_path (str): Ruta donde se guardará el preprocesador.
        metrics_path (str): Ruta donde se guardarán las métricas de entrenamiento.

    Returns:
        pipeline: Pipeline entrenado.
        preprocessor: Preprocesador entrenado.
        metrics: Diccionario con las métricas obtenidas.
    """

    # Configuración de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("13MBID - Modelo Training Pipeline")

    # Cargar los datos
    X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y = load_data(data_path)

    # Codificar la variable objetivo si es necesario
    y_train_eval, y_val_eval, y_test_eval = encode_target(y_train, y_val, y_test)

    # Crear el preprocesador
    preprocessor = create_preprocessor(features_X)

    # Modelo base
    modelo = GradientBoostingClassifier(random_state=42)

    # Pipeline completo con preprocesamiento, undersampling y modelo
    pipeline = ImbPipeline([
        ("prep", preprocessor),
        ("undersample", RandomUnderSampler(random_state=42)),
        ("model", modelo)
    ])

    # Espacio de búsqueda de hiperparámetros
    param_distributions = {
        "model__n_estimators": randint(50, 200),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_depth": randint(2, 6),
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 10),
        "model__subsample": uniform(0.7, 0.3)
    }

    # Optimización mediante RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Entrenamiento con búsqueda de hiperparámetros
    random_search.fit(X_train, y_train_eval)

    # Se recupera el mejor pipeline encontrado
    pipeline = random_search.best_estimator_

    print("Mejores parámetros encontrados:")
    print(random_search.best_params_)
    print(f"Mejor ROC-AUC en validación cruzada: {random_search.best_score_:.4f}")

    # Evaluación sobre el conjunto de test
    y_test_pred = pipeline.predict(X_test)

    y_test_score = pipeline.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(y_test_eval, y_test_score)

    metrics = {
        "test_accuracy": accuracy_score(y_test_eval, y_test_pred),
        "test_precision": precision_score(y_test_eval, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test_eval, y_test_pred, zero_division=0),
        "test_f1": f1_score(y_test_eval, y_test_pred, zero_division=0),
        "test_roc_auc": test_roc_auc,
        "best_cv_roc_auc": random_search.best_score_
    }

    print("\nMétricas en test:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Crear carpetas necesarias
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path("docs/figures").mkdir(parents=True, exist_ok=True)

    # Matriz de confusión
    cm = confusion_matrix(y_test_eval, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión - Gradient Boosting optimizado")

    cm_path = "docs/figures/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # Registro en MLflow
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name="Pipeline (prod) - GradientBoosting RandomizedSearchCV"):
        # Parámetros del mejor modelo
        mlflow.log_params(pipeline.named_steps["model"].get_params())

        # Mejores parámetros encontrados por RandomizedSearchCV
        mlflow.log_params({
            f"best_{k}": v for k, v in random_search.best_params_.items()
        })

        # Parámetros generales del entrenamiento
        mlflow.log_params({
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "balancing_method": "undersampling",
            "search_method": "RandomizedSearchCV",
            "cv": 5,
            "scoring": "roc_auc",
            "n_iter": 20
        })

        # Métricas
        mlflow.log_metrics(metrics)

        # Artefactos
        mlflow.log_artifact(cm_path)

        # Modelo
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\nModelo registrado en MLflow. run_id: {run_id}")

    # Guardar modelo completo
    joblib.dump(pipeline, model_path)

    # Guardar preprocesador entrenado
    joblib.dump(pipeline.named_steps["prep"], preprocessor_path)

    # Guardar métricas en JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModelo guardado en: {model_path}")
    print(f"Preprocesador guardado en: {preprocessor_path}")
    print(f"Métricas guardadas en: {metrics_path}")

    return pipeline, pipeline.named_steps["prep"], metrics


if __name__ == "__main__":
    train_model()