from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "default_credit.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.parquet"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
DASHBOARD_DATA_PATH = PROJECT_ROOT / "docs" / "data" / "dashboard.json"

DATASET_ID = 350
TARGET_COLUMN = "default_next_month"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CATEGORICAL_COLUMNS = (
    "sex",
    "education",
    "marriage",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
)
