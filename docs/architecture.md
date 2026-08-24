# Architecture

## Local-first stage

1. Fetch the official UCI dataset through its repository client.
2. Normalize and validate the schema.
3. Use a fixed stratified split for reproducibility.
4. Train an interpretable logistic-regression baseline and a nonlinear gradient-boosting challenger.
5. Select the model with the highest precision-recall AUC.
6. Persist model, metrics, and offline predictions.

## Controlled AWS stage

```text
Prepared CSV → S3 → SageMaker training job → model.tar.gz
                                         ↓
Batch input  → SageMaker Batch Transform → S3 predictions
                                         ↓
                                local evidence + cleanup
```

Batch inference avoids a continuously provisioned endpoint. Every cloud command will require an explicit profile and region, use tagged resources, and include a cleanup command.

### Guardrails

- AWS profile fixed to `guilherme-admin`
- Region fixed to `us-east-2`
- Private encrypted S3 bucket with seven-day lifecycle expiration
- Project-scoped IAM role and inline policy
- One `ml.m5.large` training instance, capped at 15 minutes
- One `ml.m5.large` Batch Transform instance
- No endpoint configuration or real-time endpoint
- Mutating commands require `--execute`
- Cleanup refuses to run without the exact local evidence file
