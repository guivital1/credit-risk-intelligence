"""Lightweight version registry with optional MLflow tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def register_model(
    summary: dict[str, Any],
    model_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    entry = {
        "version": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "candidate",
        "model": summary["selected_model"],
        "selection_metric": summary["selection_metric"],
        "metrics": summary["models"][summary["selected_model"]],
        "artifact_sha256": digest,
    }
    existing = json.loads(registry_path.read_text()) if registry_path.exists() else {"models": []}
    existing["models"].append(entry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return entry


def log_to_mlflow(summary: dict[str, Any], model_path: Path) -> bool:
    """Log when the optional MLflow dependency is installed; remain local-first otherwise."""

    try:
        import mlflow
    except ImportError:
        return False
    selected = summary["selected_model"]
    with mlflow.start_run(run_name=f"credit-risk-{selected}"):
        mlflow.log_param("selected_model", selected)
        mlflow.log_param("selection_metric", summary["selection_metric"])
        mlflow.log_metrics(
            {
                key: value
                for key, value in summary["models"][selected].items()
                if isinstance(value, int | float)
            }
        )
        mlflow.log_artifact(str(model_path))
    return True
