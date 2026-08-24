from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import DASHBOARD_DATA_PATH, METRICS_PATH, PREDICTIONS_PATH


def build_dashboard_payload(
    predictions: pd.DataFrame, metrics: dict[str, Any]
) -> dict[str, Any]:
    required = {"actual_default", "default_probability", "predicted_default"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    scored = predictions.copy()
    scored["risk_band"] = pd.cut(
        scored["default_probability"],
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["low", "review", "high"],
        right=False,
    )

    risk_bands = []
    for band in ("low", "review", "high"):
        subset = scored.loc[scored["risk_band"] == band]
        risk_bands.append(
            {
                "band": band,
                "clients": int(len(subset)),
                "observed_default_rate": round(float(subset["actual_default"].mean()), 6)
                if len(subset)
                else 0.0,
                "average_probability": round(float(subset["default_probability"].mean()), 6)
                if len(subset)
                else 0.0,
            }
        )

    calibration = []
    bin_index = pd.cut(
        scored["default_probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    for interval, subset in scored.groupby(bin_index, observed=False):
        if subset.empty:
            continue
        calibration.append(
            {
                "bin": str(interval),
                "clients": int(len(subset)),
                "average_probability": round(float(subset["default_probability"].mean()), 6),
                "observed_default_rate": round(float(subset["actual_default"].mean()), 6),
            }
        )

    models = [
        {"name": name, **values}
        for name, values in metrics["models"].items()
    ]
    return {
        "summary": {
            "scored_rows": int(len(scored)),
            "observed_defaults": int(scored["actual_default"].sum()),
            "default_rate": round(float(scored["actual_default"].mean()), 6),
            "average_probability": round(float(scored["default_probability"].mean()), 6),
            "selected_model": metrics["selected_model"],
            "selection_metric": metrics["selection_metric"],
        },
        "models": models,
        "risk_bands": risk_bands,
        "calibration": calibration,
        "disclaimer": (
            "Educational portfolio project using a public research dataset. "
            "Not for real credit decisions."
        ),
    }


def generate_dashboard_data(
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
    destination: Path = DASHBOARD_DATA_PATH,
) -> dict[str, Any]:
    predictions = pd.read_parquet(predictions_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = build_dashboard_payload(predictions, metrics)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

