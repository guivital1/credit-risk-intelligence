from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import (
    CATEGORICAL_COLUMNS,
    METRICS_PATH,
    MODEL_CARD_PATH,
    MODEL_PATH,
    MONITORING_PATH,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    REGISTRY_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from credit_risk.governance import evaluate_promotion
from credit_risk.monitoring import feature_drift_report, prediction_drift, slice_performance
from credit_risk.registry import log_to_mlflow, register_model


@dataclass(frozen=True)
class TrainingResult:
    selected_model: str
    metrics: dict[str, dict[str, Any]]
    test_rows: int
    default_rate: float


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return frame.drop(columns=TARGET_COLUMN), frame[TARGET_COLUMN].astype(int)


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in columns]
    numeric = [column for column in columns if column not in categorical]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        verbose_feature_names_out=False,
    )


def build_models(columns: list[str]) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(columns)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor(columns)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.07,
                        max_iter=180,
                        max_leaf_nodes=24,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def classification_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 6),
        "pr_auc": round(float(average_precision_score(y_true, probability)), 6),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 6),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 6),
        "confusion_matrix": matrix.tolist(),
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    monitoring_path: Path = MONITORING_PATH,
    registry_path: Path = REGISTRY_PATH,
    model_card_path: Path = MODEL_CARD_PATH,
) -> TrainingResult:
    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = build_models(features.columns.tolist())
    metrics: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        fitted[name] = pipeline
        probabilities[name] = probability
        metrics[name] = classification_metrics(y_test, probability)

    selected_name = max(metrics, key=lambda name: metrics[name]["pr_auc"])
    selected_model = fitted[selected_name]

    importance = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    feature_importance = sorted(
        (
            {"feature": column, "importance": round(float(value), 6)}
            for column, value in zip(x_test.columns, importance.importances_mean, strict=True)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )[:10]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, model_path)

    summary = {
        "selection_metric": "pr_auc",
        "selected_model": selected_name,
        "test_rows": len(x_test),
        "test_default_rate": round(float(y_test.mean()), 6),
        "models": metrics,
        "explainability": {
            "method": "permutation importance on the held-out test split",
            "top_features": feature_importance,
        },
        "promotion": evaluate_promotion(
            metrics["logistic_regression"], metrics["gradient_boosting"]
        ),
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_probability = selected_model.predict_proba(x_train)[:, 1]
    test_probability = probabilities[selected_name]
    numeric_monitoring_columns = [
        column for column in ("limit_bal", "age", "bill_amt1", "pay_amt1") if column in x_train
    ]
    monitoring = {
        "prediction_drift": prediction_drift(train_probability, test_probability),
        "feature_drift": feature_drift_report(
            x_train, x_test, numeric_monitoring_columns
        ),
        "governance_slices": {
            column: slice_performance(x_test, y_test, test_probability, column)
            for column in ("sex", "education", "marriage")
            if column in x_test
        },
        "thresholds": {"watch_psi": 0.10, "alert_psi": 0.25},
    }
    monitoring_path.write_text(json.dumps(monitoring, indent=2), encoding="utf-8")

    predictions = x_test.copy()
    predictions["actual_default"] = y_test.to_numpy()
    predictions["default_probability"] = probabilities[selected_name]
    predictions["predicted_default"] = (probabilities[selected_name] >= 0.5).astype(int)
    predictions.to_parquet(predictions_path, index=False)

    registry_entry = register_model(summary, model_path, registry_path)
    mlflow_logged = log_to_mlflow(summary, model_path)
    selected_metrics = metrics[selected_name]
    model_card_path.write_text(
        "\n".join(
            [
                "# Credit Risk Intelligence — Model Card",
                "",
                f"- Version: `{registry_entry['version']}`",
                f"- Model: `{selected_name}`",
                f"- ROC-AUC: {selected_metrics['roc_auc']}",
                f"- PR-AUC: {selected_metrics['pr_auc']}",
                f"- Brier score: {selected_metrics['brier_score']}",
                f"- MLflow logged: {mlflow_logged}",
                "",
                "## Intended use",
                "Educational offline decision support and model-governance demonstration.",
                "It must not approve, deny or price real credit products.",
                "",
                "## Monitoring",
                "Prediction and feature PSI, calibration, PR-AUC and governance slices must be",
                "reviewed before promotion. PSI >= 0.25 blocks promotion and triggers",
                "investigation.",
                "",
                "## Limitations",
                "The public research dataset is historical and does not represent a production",
                "Brazilian lending population. Slice reports diagnose behavior; they do not",
                "prove fairness.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return TrainingResult(
        selected_model=selected_name,
        metrics=metrics,
        test_rows=len(x_test),
        default_rate=float(y_test.mean()),
    )
