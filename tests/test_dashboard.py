import pandas as pd

from credit_risk.dashboard import build_dashboard_payload


def test_build_dashboard_payload_groups_risk_bands() -> None:
    predictions = pd.DataFrame(
        {
            "actual_default": [0, 0, 1, 1],
            "default_probability": [0.05, 0.25, 0.55, 0.85],
            "predicted_default": [0, 0, 1, 1],
        }
    )
    metrics = {
        "selected_model": "gradient_boosting",
        "selection_metric": "pr_auc",
        "models": {
            "gradient_boosting": {"roc_auc": 0.8, "pr_auc": 0.6},
        },
    }
    payload = build_dashboard_payload(predictions, metrics)
    assert payload["summary"]["scored_rows"] == 4
    assert [band["clients"] for band in payload["risk_bands"]] == [1, 1, 2]

