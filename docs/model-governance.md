# Model governance and monitoring

The workflow treats model selection as an auditable promotion decision rather
than simply choosing the largest score.

## Registry and experiment tracking

Every training run creates a candidate entry with metrics, timestamp and the
SHA-256 digest of the serialized artifact. The local JSON registry is always
available. Installing the optional `mlops` dependency also records parameters,
metrics and the artifact in MLflow.

## Promotion policy

The gradient-boosting challenger must:

- improve PR-AUC by at least 0.005;
- keep Brier-score degradation below 0.01;
- maintain recall of at least 0.30.

A failed check blocks promotion. Production rollback should be triggered when
PR-AUC or calibration breaches the approved threshold.

## Drift

Population Stability Index is calculated for predictions and selected numeric
features. PSI below 0.10 is stable, 0.10–0.25 requires review and 0.25 or above
is an alert. Bins are derived only from the reference population.

## Responsible slices

Offline reports compare observed risk, predicted risk and PR-AUC across sex,
education and marital-status codes present in the research dataset. These
diagnostics can reveal inconsistent behavior but do not prove legal or ethical
fairness and must not be used to justify real credit decisions.

## Explainability

Permutation importance is calculated on the held-out test split. It is
model-agnostic and measures predictive impact, not causality. The generated
model card records intended use, limitations and monitoring expectations.
