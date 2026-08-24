from pathlib import Path

import pandas as pd

from credit_risk.aws_data import prepare_sagemaker_data
from credit_risk.config import TARGET_COLUMN


def test_prepare_sagemaker_data_writes_label_first_without_header(tmp_path: Path) -> None:
    rows = 40
    frame = pd.DataFrame(
        {
            "limit_bal": range(1, rows + 1),
            "age": range(20, 20 + rows),
            TARGET_COLUMN: [0, 1] * (rows // 2),
        }
    )
    manifest = prepare_sagemaker_data(frame, tmp_path)
    first_row = (tmp_path / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first_row.split(",")[0] in {"0", "1"}
    assert "default_next_month" not in first_row
    assert manifest["feature_count"] == 2
    assert sum(manifest["rows"].values()) == rows

