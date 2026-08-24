"""Explicit model-promotion policy for reproducible challenger decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_pr_auc_gain: float = 0.005
    maximum_brier_degradation: float = 0.01
    minimum_recall: float = 0.30


def evaluate_promotion(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    policy: PromotionPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or PromotionPolicy()
    checks = {
        "pr_auc_gain": challenger["pr_auc"] - champion["pr_auc"] >= policy.minimum_pr_auc_gain,
        "calibration": (
            challenger["brier_score"] - champion["brier_score"]
            <= policy.maximum_brier_degradation
        ),
        "minimum_recall": challenger["recall"] >= policy.minimum_recall,
    }
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "policy": asdict(policy),
        "rollback_trigger": "PR-AUC or calibration breaches the approved production threshold",
    }
