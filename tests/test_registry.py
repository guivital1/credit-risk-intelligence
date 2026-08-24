import json

from credit_risk.registry import register_model


def test_registry_versions_model_with_artifact_digest(tmp_path) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes(b"deterministic-model")
    registry = tmp_path / "registry.json"
    summary = {
        "selected_model": "gradient_boosting",
        "selection_metric": "pr_auc",
        "models": {"gradient_boosting": {"pr_auc": 0.55}},
    }
    entry = register_model(summary, model, registry)
    assert len(entry["artifact_sha256"]) == 64
    assert json.loads(registry.read_text())["models"][0]["stage"] == "candidate"
