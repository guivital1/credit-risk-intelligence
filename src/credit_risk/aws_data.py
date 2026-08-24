from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import RANDOM_STATE, TARGET_COLUMN
from credit_risk.modeling import split_features_target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_labeled(path: Path, features: pd.DataFrame, target: pd.Series) -> None:
    output = pd.concat(
        [target.rename(TARGET_COLUMN).reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    output.to_csv(path, index=False, header=False)


def prepare_sagemaker_data(
    frame: pd.DataFrame,
    destination: Path,
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Create SageMaker XGBoost CSV channels with label-first, headerless rows."""
    features, target = split_features_target(frame)
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=random_state,
        stratify=target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        random_state=random_state,
        stratify=y_holdout,
    )

    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": destination / "train.csv",
        "validation": destination / "validation.csv",
        "transform": destination / "transform.csv",
        "test_labels": destination / "test_labels.csv",
    }
    _write_labeled(paths["train"], x_train, y_train)
    _write_labeled(paths["validation"], x_validation, y_validation)
    x_test.to_csv(paths["transform"], index=False, header=False)
    y_test.rename(TARGET_COLUMN).to_csv(paths["test_labels"], index=False, header=False)

    manifest = {
        "feature_columns": features.columns.tolist(),
        "feature_count": len(features.columns),
        "rows": {
            "train": len(x_train),
            "validation": len(x_validation),
            "transform": len(x_test),
        },
        "default_rates": {
            "train": round(float(y_train.mean()), 6),
            "validation": round(float(y_validation.mean()), 6),
            "transform": round(float(y_test.mean()), 6),
        },
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest

