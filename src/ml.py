"""
ML layer: train regression and classification models on gold feature table.
Logs everything to MLflow: params, metrics, model artifacts, feature importance,
and the Delta version of the gold table used for training.
"""

import logging

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deltalake import DeltaTable
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import GOLD_FEATURES_PATH, MLFLOW_URI, EXPERIMENT_NAME, DELAY_THRESHOLD_MIN

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = ["IATA_Code_Operating_Airline", "Origin", "Dest", "route", "season"]
NUMERIC_COLS = [
    "month",
    "day_of_week",
    "hour",
    "Distance",
    "DepDelay",
    "TaxiOut",
    "CRSElapsedTime",
]
TARGET_REG = "ArrDelay"
TARGET_CLF = "is_delayed"


def load_features() -> tuple[pl.DataFrame, int]:
    dt = DeltaTable(GOLD_FEATURES_PATH)
    gold_version = dt.version()
    df = pl.scan_delta(GOLD_FEATURES_PATH).collect()
    return df, gold_version


def encode_features(df: pl.DataFrame) -> tuple[np.ndarray, dict]:
    encoders = {}
    pdf = df.to_pandas()
    for col in CATEGORICAL_COLS:
        if col in pdf.columns:
            le = LabelEncoder()
            pdf[col] = le.fit_transform(pdf[col].astype(str))
            encoders[col] = le
    feature_cols = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c in pdf.columns]
    X = pdf[feature_cols].fillna(0).values
    return X, feature_cols, encoders


def plot_feature_importance(
    model, feature_cols: list[str], title: str, path: str
) -> None:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    idx = np.argsort(importance)[-20:]
    ax.barh([feature_cols[i] for i in idx], importance[idx])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_regression(X_train, X_test, y_train, y_test, feature_cols, gold_version):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = {
        "lgbm_regressor": LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=63, random_state=42
        ),
        "linear_regression": LinearRegression(),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"reg_{name}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = mean_squared_error(y_test, preds) ** 0.5
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            mlflow.log_params(
                {
                    "model": name,
                    "task": "regression",
                    "target": TARGET_REG,
                    "gold_table_version": gold_version,
                    "delay_threshold_min": DELAY_THRESHOLD_MIN,
                }
            )
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            if "lgbm" in name:
                mlflow.lightgbm.log_model(model, artifact_path="model")
                imp_path = f"logs/fi_reg_{name}.png"
                plot_feature_importance(
                    model, feature_cols, f"Feature Importance: {name}", imp_path
                )
                mlflow.log_artifact(imp_path, artifact_path="plots")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            logger.info(f"[REG] {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")


def train_classification(X_train, X_test, y_train, y_test, feature_cols, gold_version):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = {
        "lgbm_classifier": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63, random_state=42
        ),
        "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"clf_{name}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else preds
            )

            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, proba)

            mlflow.log_params(
                {
                    "model": name,
                    "task": "classification",
                    "target": TARGET_CLF,
                    "gold_table_version": gold_version,
                    "delay_threshold_min": DELAY_THRESHOLD_MIN,
                }
            )
            mlflow.log_metrics({"f1": f1, "roc_auc": auc})

            if "lgbm" in name:
                mlflow.lightgbm.log_model(model, artifact_path="model")
                imp_path = f"logs/fi_clf_{name}.png"
                plot_feature_importance(
                    model, feature_cols, f"Feature Importance: {name}", imp_path
                )
                mlflow.log_artifact(imp_path, artifact_path="plots")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            logger.info(f"[CLF] {name}: F1={f1:.3f}, AUC={auc:.3f}")


def run_ml() -> None:
    logger.info("Loading gold feature table…")
    df, gold_version = load_features()
    logger.info(f"Feature table: {len(df):,} rows, gold version={gold_version}")

    X, feature_cols, _ = encode_features(df)
    y_reg = df[TARGET_REG].to_numpy()
    y_clf = df[TARGET_CLF].to_numpy()

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    logger.info("Training regression models…")
    train_regression(X_train, X_test, yr_train, yr_test, feature_cols, gold_version)

    logger.info("Training classification models…")
    train_classification(X_train, X_test, yc_train, yc_test, feature_cols, gold_version)

    logger.info("ML complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ml()
