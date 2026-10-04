"""Typed view over the `laya` block in cdk.json (override with `cdk deploy -c laya='{...}'`)."""

from dataclasses import dataclass

from constructs import Construct


@dataclass(frozen=True)
class LayaConfig:
    endpoint_name: str
    memory_size_in_mb: int
    max_concurrency: int
    provisioned_concurrency: int
    laya_subfolder: str
    bedrock_model_id: str
    confidence_threshold: float
    notification_email: str

    @classmethod
    def from_context(cls, scope: Construct) -> "LayaConfig":
        raw = scope.node.try_get_context("laya") or {}
        cfg = cls(
            endpoint_name=raw.get("endpointName", "laya-serverless"),
            memory_size_in_mb=int(raw.get("memorySizeInMb", 6144)),
            max_concurrency=int(raw.get("maxConcurrency", 5)),
            provisioned_concurrency=int(raw.get("provisionedConcurrency", 0)),
            laya_subfolder=raw.get("layaSubfolder", ""),
            bedrock_model_id=raw.get("bedrockModelId", "eu.anthropic.claude-haiku-4-5-20251001-v1:0"),
            confidence_threshold=float(raw.get("confidenceThreshold", 0.5)),
            notification_email=raw.get("notificationEmail", ""),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        # SageMaker Serverless limits: 1024-6144 MB in 1 GB steps, concurrency 1-200.
        if self.memory_size_in_mb not in range(1024, 6145, 1024):
            raise ValueError("memorySizeInMb must be one of 1024, 2048, ..., 6144")
        if not 1 <= self.max_concurrency <= 200:
            raise ValueError("maxConcurrency must be between 1 and 200")
        if not 0 <= self.provisioned_concurrency <= self.max_concurrency:
            raise ValueError("provisionedConcurrency must be between 0 and maxConcurrency")
        if self.laya_subfolder not in ("", "multilingual", "typed-decisions"):
            raise ValueError("layaSubfolder must be '', 'multilingual' or 'typed-decisions'")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidenceThreshold must be between 0 and 1")


@dataclass(frozen=True)
class StrandsDeciderConfig:
    """`strandsDecider` block in cdk.json: the second serverless endpoint."""

    endpoint_name: str
    memory_size_in_mb: int
    max_concurrency: int
    provisioned_concurrency: int
    checkpoint_repo: str

    @classmethod
    def from_context(cls, scope: Construct) -> "StrandsDeciderConfig":
        raw = scope.node.try_get_context("strandsDecider") or {}
        cfg = cls(
            endpoint_name=raw.get("endpointName", "strands-decider-serverless"),
            memory_size_in_mb=int(raw.get("memorySizeInMb", 6144)),
            max_concurrency=int(raw.get("maxConcurrency", 2)),
            provisioned_concurrency=int(raw.get("provisionedConcurrency", 0)),
            checkpoint_repo=raw.get("checkpointRepo", "StrandsAgents/strands-decider-2B-hobson-v19"),
        )
        if cfg.memory_size_in_mb not in range(1024, 6145, 1024):
            raise ValueError("strandsDecider.memorySizeInMb must be one of 1024, 2048, ..., 6144")
        if not 1 <= cfg.max_concurrency <= 200:
            raise ValueError("strandsDecider.maxConcurrency must be between 1 and 200")
        if not 0 <= cfg.provisioned_concurrency <= cfg.max_concurrency:
            raise ValueError("strandsDecider.provisionedConcurrency must be between 0 and maxConcurrency")
        return cfg
