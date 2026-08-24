from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from credit_risk.config import DATASET_ID, RAW_DATA_PATH, TARGET_COLUMN


def normalize_column_name(value: object) -> str:
    """Convert source column names into stable snake_case names."""
    name = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    aliases = {
        "default_payment_next_month": TARGET_COLUMN,
        "default_payment_next_month_": TARGET_COLUMN,
        "y": TARGET_COLUMN,
    }
    return aliases.get(name, name)


def normalize_dataset(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join UCI features and target while enforcing a stable schema."""
    frame = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    frame.columns = [normalize_column_name(column) for column in frame.columns]

    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate normalized columns: {duplicate_columns}")

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Expected target column {TARGET_COLUMN!r}; found {frame.columns.tolist()}"
        )

    if "id" in frame.columns:
        frame = frame.drop(columns="id")

    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype("int8")
    if set(frame[TARGET_COLUMN].unique()) - {0, 1}:
        raise ValueError("Target must contain only 0 and 1")
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Dataset contains missing values in: {missing}")
    return frame


def fetch_dataset(destination: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Download the official UCI dataset and persist a normalized CSV copy."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    frame = normalize_dataset(dataset.data.features, dataset.data.targets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def load_dataset(path: Path = RAW_DATA_PATH, *, fetch_if_missing: bool = True) -> pd.DataFrame:
    """Load the normalized local dataset, optionally fetching it first."""
    if not path.exists():
        if not fetch_if_missing:
            raise FileNotFoundError(path)
        return fetch_dataset(path)
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return frame
