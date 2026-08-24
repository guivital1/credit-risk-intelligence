# Credit Risk Intelligence

An explainable machine-learning project for estimating credit-card default risk. The project starts with a fully reproducible local workflow and later mirrors the selected model in a short, cost-controlled Amazon SageMaker training job.

## Decision supported

The model estimates the probability that a client will default in the following month. It is a portfolio demonstration, not an automated credit-decision system. Predictions must not be used to approve, deny, or price real financial products.

## Dataset

The project uses **Default of Credit Card Clients**, published by the UCI Machine Learning Repository:

- 30,000 records
- 23 predictive variables
- Binary classification target
- CC BY 4.0 license
- DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)

Suggested citation: Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository.

## Local workflow

```text
UCI dataset
    ↓
schema normalization + validation
    ↓
stratified train/test split
    ↓
logistic regression ↔ gradient boosting
    ↓
ROC-AUC + PR-AUC + recall + calibration
    ↓
model artifact + batch predictions + dashboard snapshot
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
credit-risk all
pytest
```

Preview the AWS plan without creating resources:

```bash
python -m pip install -e '.[aws,dev]'
credit-risk-cloud plan
```

Cloud mutations require an explicit safeguard flag:

```bash
credit-risk-cloud deploy --execute
credit-risk-cloud cleanup --execute
```

Generated files are intentionally ignored by Git:

- `data/raw/default_credit.csv`
- `data/processed/test_predictions.parquet`
- `artifacts/model.joblib`
- `artifacts/metrics.json`

## AWS design

The cloud stage will upload prepared data to Amazon S3, launch a single managed SageMaker training job, run offline Batch Transform inference, download the evidence, and remove temporary resources. No persistent inference endpoint is required.

## Responsible-use boundaries

- No real customer or banking data
- No protected attributes used for a real decision
- Metrics reported beyond accuracy because defaults are imbalanced
- Model output treated as decision support, never as an automatic verdict
- Cloud resources created only during controlled demonstrations
