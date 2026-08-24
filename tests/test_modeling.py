import numpy as np
import pandas as pd

from credit_risk.config import TARGET_COLUMN
from credit_risk.modeling import classification_metrics, split_features_target


def test_split_features_target() -> None:
    frame = pd.DataFrame({"limit_bal": [1, 2], TARGET_COLUMN: [0, 1]})
    features, target = split_features_target(frame)
    assert features.columns.tolist() == ["limit_bal"]
    assert target.tolist() == [0, 1]


def test_classification_metrics_contains_risk_metrics() -> None:
    target = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.7, 0.9])
    metrics = classification_metrics(target, probability)
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]

