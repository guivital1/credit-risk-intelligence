from credit_risk.governance import PromotionPolicy, evaluate_promotion


def metrics(pr_auc: float, brier: float, recall: float) -> dict[str, float]:
    return {"pr_auc": pr_auc, "brier_score": brier, "recall": recall}


def test_better_calibrated_challenger_is_promoted() -> None:
    result = evaluate_promotion(metrics(0.50, 0.20, 0.40), metrics(0.56, 0.15, 0.35))
    assert result["promote"] is True


def test_recall_floor_blocks_promotion() -> None:
    policy = PromotionPolicy(minimum_recall=0.30)
    result = evaluate_promotion(
        metrics(0.50, 0.20, 0.40), metrics(0.56, 0.15, 0.10), policy
    )
    assert result["promote"] is False
    assert result["checks"]["minimum_recall"] is False
