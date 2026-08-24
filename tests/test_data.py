import pandas as pd
import pytest

from credit_risk.config import TARGET_COLUMN
from credit_risk.data import load_dataset, normalize_column_name, normalize_dataset


def test_normalize_column_name_maps_target() -> None:
    assert normalize_column_name("default payment next month") == TARGET_COLUMN
    assert normalize_column_name("LIMIT BAL") == "limit_bal"
    assert normalize_column_name("X1") == "limit_bal"
    assert normalize_column_name("X23") == "pay_amt6"


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


def test_load_dataset_upgrades_generic_uci_columns(tmp_path) -> None:
    path = tmp_path / "credit.csv"
    pd.DataFrame(
        {"x1": [10_000], "x2": [2], "default_next_month": [0]}
    ).to_csv(path, index=False)

    result = load_dataset(path, fetch_if_missing=False)

    assert result.columns.tolist() == ["limit_bal", "sex", TARGET_COLUMN]
