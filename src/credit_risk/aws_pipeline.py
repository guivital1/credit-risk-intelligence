from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_risk.aws_data import prepare_sagemaker_data
from credit_risk.config import PROJECT_ROOT
from credit_risk.data import load_dataset

PROFILE = "guilherme-admin"
REGION = "us-east-2"
IMAGE_URI = "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.7-1"
PROJECT_TAG = "credit-risk-intelligence"
ROLE_NAME = "CreditRiskSageMakerExecutionRole"
INLINE_POLICY_NAME = "CreditRiskSageMakerProjectPolicy"
LOCAL_AWS_DIR = PROJECT_ROOT / "data" / "processed" / "sagemaker"
EVIDENCE_PATH = PROJECT_ROOT / "artifacts" / "aws-evidence.json"
AWS_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "aws_predictions.csv"


@dataclass(frozen=True)
class CloudContext:
    session: boto3.Session
    account_id: str
    bucket: str
    role_arn: str
    role_created: bool


def _session() -> boto3.Session:
    return boto3.Session(profile_name=PROFILE, region_name=REGION)


def get_identity() -> dict[str, str]:
    identity = _session().client("sts").get_caller_identity()
    return {"account": identity["Account"], "arn": identity["Arn"]}


def prepare_local_channels() -> dict[str, Any]:
    return prepare_sagemaker_data(load_dataset(), LOCAL_AWS_DIR)


def plan() -> dict[str, Any]:
    identity = get_identity()
    manifest = prepare_local_channels()
    bucket = f"credit-risk-intelligence-{identity['account']}-{REGION}"
    return {
        "identity": identity,
        "region": REGION,
        "bucket": bucket,
        "role": ROLE_NAME,
        "training_instance": "ml.m5.large",
        "training_max_seconds": 900,
        "transform_instance": "ml.m5.large",
        "transform_max_seconds": 600,
        "persistent_endpoint": False,
        "automatic_cleanup": True,
        "dataset": manifest,
    }


def _ensure_bucket(session: boto3.Session, bucket: str) -> None:
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                    "BucketKeyEnabled": True,
                }
            ]
        },
    )
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-demo-artifacts",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "runs/"},
                    "Expiration": {"Days": 7},
                }
            ]
        },
    )


def _ensure_role(session: boto3.Session, bucket: str) -> tuple[str, bool]:
    iam = session.client("iam")
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    created = False
    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        role_tags = iam.list_role_tags(RoleName=ROLE_NAME)["Tags"]
        tags = {item["Key"]: item["Value"] for item in role_tags}
        if tags.get("Project") != PROJECT_TAG:
            raise RuntimeError(f"Existing role {ROLE_NAME} is not owned by this project")
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Temporary execution role for the Credit Risk Intelligence demo",
            MaxSessionDuration=3600,
            Tags=[
                {"Key": "Project", "Value": PROJECT_TAG},
                {"Key": "AutoCleanup", "Value": "true"},
            ],
        )["Role"]
        created = True

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ProjectBucketList",
                "Effect": "Allow",
                "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
            {
                "Sid": "ProjectBucketObjects",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {
                "Sid": "ManagedContainerRead",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                "Resource": "*",
            },
            {
                "Sid": "TrainingObservability",
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:DescribeLogStreams",
                    "logs:PutLogEvents",
                ],
                "Resource": "*",
            },
        ],
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName=INLINE_POLICY_NAME,
        PolicyDocument=json.dumps(policy),
    )
    return role["Arn"], created


def _context() -> CloudContext:
    session = _session()
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket = f"credit-risk-intelligence-{account_id}-{REGION}"
    _ensure_bucket(session, bucket)
    role_arn, role_created = _ensure_role(session, bucket)
    return CloudContext(session, account_id, bucket, role_arn, role_created)


def _upload_channels(context: CloudContext, run_prefix: str) -> dict[str, str]:
    manifest = prepare_local_channels()
    s3 = context.session.client("s3")
    uris: dict[str, str] = {}
    for name in ("train", "validation", "transform"):
        path = Path(manifest["files"][name]["path"])
        key = f"{run_prefix}/input/{path.name}"
        s3.upload_file(
            str(path),
            context.bucket,
            key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        uris[name] = f"s3://{context.bucket}/{key}"
    return uris


def _wait_for_job(
    describe: Any,
    name_key: str,
    name: str,
    status_key: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        description = describe(**{name_key: name})
        status = description[status_key]
        if status in {"Completed", "Failed", "Stopped"}:
            if status != "Completed":
                reason = description.get("FailureReason", "unknown failure")
                raise RuntimeError(f"{name} ended with {status}: {reason}")
            return description
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {name}")
        time.sleep(20)


def deploy_and_run() -> dict[str, Any]:
    context = _context()
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"credit-risk-{timestamp}"
    run_prefix = f"runs/{run_id}"
    model_name = f"{run_id}-model"
    transform_name = f"{run_id}-batch"
    evidence = {
        "run_id": run_id,
        "region": REGION,
        "bucket": context.bucket,
        "role_name": ROLE_NAME,
        "role_created": context.role_created,
        "training_job": run_id,
        "model_name": model_name,
        "transform_job": transform_name,
        "status": "resources-prepared",
        "persistent_endpoint": False,
        "cleanup_complete": False,
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    uris = _upload_channels(context, run_prefix)
    sagemaker = context.session.client("sagemaker")
    time.sleep(10 if context.role_created else 0)
    sagemaker.create_training_job(
        TrainingJobName=run_id,
        AlgorithmSpecification={
            "TrainingImage": IMAGE_URI,
            "TrainingInputMode": "File",
        },
        RoleArn=context.role_arn,
        InputDataConfig=[
            {
                "ChannelName": channel,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": uris[channel],
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
                "CompressionType": "None",
            }
            for channel in ("train", "validation")
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{context.bucket}/{run_prefix}/model"},
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 900},
        EnableNetworkIsolation=True,
        HyperParameters={
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "num_round": "180",
            "max_depth": "5",
            "eta": "0.07",
            "subsample": "0.8",
            "colsample_bytree": "0.8",
            "min_child_weight": "5",
        },
        Tags=[
            {"Key": "Project", "Value": PROJECT_TAG},
            {"Key": "AutoCleanup", "Value": "true"},
        ],
    )
    training = _wait_for_job(
        sagemaker.describe_training_job,
        "TrainingJobName",
        run_id,
        "TrainingJobStatus",
        timeout_seconds=1_200,
    )

    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": training["ModelArtifacts"]["S3ModelArtifacts"],
        },
        ExecutionRoleArn=context.role_arn,
        EnableNetworkIsolation=True,
        Tags=[
            {"Key": "Project", "Value": PROJECT_TAG},
            {"Key": "AutoCleanup", "Value": "true"},
        ],
    )

    transform_output = f"s3://{context.bucket}/{run_prefix}/transform-output"
    sagemaker.create_transform_job(
        TransformJobName=transform_name,
        ModelName=model_name,
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=6,
        BatchStrategy="MultiRecord",
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uris["transform"]}},
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": transform_output,
            "Accept": "text/csv",
            "AssembleWith": "Line",
        },
        TransformResources={"InstanceType": "ml.m5.large", "InstanceCount": 1},
        Tags=[
            {"Key": "Project", "Value": PROJECT_TAG},
            {"Key": "AutoCleanup", "Value": "true"},
        ],
    )
    transform = _wait_for_job(
        sagemaker.describe_transform_job,
        "TransformJobName",
        transform_name,
        "TransformJobStatus",
        timeout_seconds=900,
    )

    output_key = f"{run_prefix}/transform-output/transform.csv.out"
    output_path = LOCAL_AWS_DIR / "transform.csv.out"
    context.session.client("s3").download_file(context.bucket, output_key, str(output_path))
    probability = pd.read_csv(output_path, header=None).iloc[:, 0]
    labels = pd.read_csv(LOCAL_AWS_DIR / "test_labels.csv", header=None).iloc[:, 0]
    if len(probability) != len(labels):
        raise RuntimeError("Batch prediction count does not match test labels")

    prediction_frame = pd.DataFrame({"actual_default": labels, "default_probability": probability})
    prediction_frame.to_csv(AWS_PREDICTIONS_PATH, index=False)
    transform_started = transform.get("TransformStartTime")
    transform_ended = transform.get("TransformEndTime")
    transform_seconds = None
    if transform_started and transform_ended:
        transform_seconds = round((transform_ended - transform_started).total_seconds())
    evidence = {
        "run_id": run_id,
        "region": REGION,
        "bucket": context.bucket,
        "role_name": ROLE_NAME,
        "role_created": context.role_created,
        "training_job": run_id,
        "training_status": training["TrainingJobStatus"],
        "training_seconds": training.get("TrainingTimeInSeconds"),
        "model_name": model_name,
        "transform_job": transform_name,
        "transform_status": transform["TransformJobStatus"],
        "transform_seconds": transform_seconds,
        "test_rows": len(labels),
        "roc_auc": round(float(roc_auc_score(labels, probability)), 6),
        "pr_auc": round(float(average_precision_score(labels, probability)), 6),
        "persistent_endpoint": False,
        "cleanup_complete": False,
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return cleanup()


def cleanup() -> dict[str, Any]:
    if not EVIDENCE_PATH.exists():
        raise FileNotFoundError("AWS evidence file not found; refusing broad cleanup")
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    session = _session()
    sagemaker = session.client("sagemaker")
    s3 = session.resource("s3")
    iam = session.client("iam")

    try:
        sagemaker.delete_model(ModelName=evidence["model_name"])
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "ValidationException":
            raise

    bucket = s3.Bucket(evidence["bucket"])
    bucket.objects.all().delete()
    bucket.delete()

    if evidence.get("role_created"):
        iam.delete_role_policy(RoleName=ROLE_NAME, PolicyName=INLINE_POLICY_NAME)
        iam.delete_role(RoleName=ROLE_NAME)

    evidence["cleanup_complete"] = True
    evidence["cleanup_at"] = datetime.now(UTC).isoformat()
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence
