"""Offline monitoring metrics for data, predictions and model slices."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Calculate PSI with quantile bins derived only from the reference sample."""

    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if not len(reference_values) or not len(current_values):
        raise ValueError("reference and current samples must not be empty")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference_values[0], current_values) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_hist = np.histogram(reference_values, bins=edges)[0] / len(reference_values)
    current_hist = np.histogram(current_values, bins=edges)[0] / len(current_values)
    epsilon = 1e-6
    psi = np.sum(
        (current_hist - reference_hist)
        * np.log((current_hist + epsilon) / (reference_hist + epsilon))
    )
    return round(float(psi), 6)


def prediction_drift(
    reference_probability: np.ndarray,
    current_probability: np.ndarray,
) -> dict[str, Any]:
    psi = population_stability_index(reference_probability, current_probability)
    return {
        "psi": psi,
        "reference_mean": round(float(np.mean(reference_probability)), 6),
        "current_mean": round(float(np.mean(current_probability)), 6),
        "status": "alert" if psi >= 0.25 else "watch" if psi >= 0.10 else "stable",
    }


def slice_performance(
    frame: pd.DataFrame,
    target: pd.Series,
    probability: np.ndarray,
    column: str,
) -> list[dict[str, Any]]:
    """Report PR-AUC and observed/predicted risk by a governance slice."""

    if column not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    aligned_target = target.reset_index(drop=True)
    aligned_frame = frame.reset_index(drop=True)
    for value, index in aligned_frame.groupby(column, dropna=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        group_target = aligned_target.iloc[positions]
        group_probability = probability[positions]
        pr_auc = (
            float(average_precision_score(group_target, group_probability))
            if group_target.nunique() > 1
            else None
        )
        rows.append(
            {
                "value": str(value),
                "rows": len(positions),
                "observed_default_rate": round(float(group_target.mean()), 6),
                "predicted_default_rate": round(float(np.mean(group_probability)), 6),
                "pr_auc": (
                    round(pr_auc, 6)
                    if pr_auc is not None and math.isfinite(pr_auc)
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda row: row["value"])


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in reference or column not in current:
            continue
        psi = population_stability_index(reference[column], current[column])
        report[column] = {
            "psi": psi,
            "status": "alert" if psi >= 0.25 else "watch" if psi >= 0.10 else "stable",
        }
    return report
