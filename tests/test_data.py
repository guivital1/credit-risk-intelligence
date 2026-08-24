import pandas as pd
import pytest

from credit_risk.config import TARGET_COLUMN
from credit_risk.data import normalize_column_name, normalize_dataset


def test_normalize_column_name_maps_target() -> None:
    assert normalize_column_name("default payment next month") == TARGET_COLUMN
    assert normalize_column_name("LIMIT BAL") == "limit_bal"


def test_normalize_dataset_joins_and_drops_id() -> None:
    features = pd.DataFrame({"ID": [1, 2], "LIMIT_BAL": [10_000, 20_000]})
    target = pd.DataFrame({"default payment next month": [0, 1]})
    result = normalize_dataset(features, target)
    assert result.columns.tolist() == ["limit_bal", TARGET_COLUMN]
    assert result[TARGET_COLUMN].tolist() == [0, 1]


def test_normalize_dataset_rejects_missing_values() -> None:
    features = pd.DataFrame({"LIMIT_BAL": [10_000, None]})
    target = pd.DataFrame({"Y": [0, 1]})
    with pytest.raises(ValueError, match="missing values"):
        normalize_dataset(features, target)

