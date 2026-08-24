import numpy as np
import pandas as pd

from credit_risk.monitoring import (
    population_stability_index,
    prediction_drift,
    slice_performance,
)


def test_identical_distribution_has_zero_psi() -> None:
    values = np.linspace(0, 1, 100)
    assert population_stability_index(values, values) == 0.0


def test_shifted_predictions_trigger_alert() -> None:
    reference = np.linspace(0.01, 0.30, 200)
    current = np.linspace(0.70, 0.99, 200)
    assert prediction_drift(reference, current)["status"] == "alert"


def test_slice_performance_reports_each_segment() -> None:
    frame = pd.DataFrame({"sex": [1, 1, 2, 2]})
    target = pd.Series([0, 1, 0, 1])
    probability = np.array([0.1, 0.8, 0.2, 0.9])
    report = slice_performance(frame, target, probability, "sex")
    assert [row["value"] for row in report] == ["1", "2"]
    assert all(row["pr_auc"] == 1.0 for row in report)
