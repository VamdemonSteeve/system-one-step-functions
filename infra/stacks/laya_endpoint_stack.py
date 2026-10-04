"""SageMaker Serverless Inference endpoint serving convaiinnovations/laya."""

from pathlib import Path

from aws_cdk import ArnFormat, CfnOutput, Stack
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

from infra.config import LayaConfig

CONTAINER_DIR = Path(__file__).resolve().parents[2] / "container"


class LayaEndpointStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, config: LayaConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Image with torch (CPU), laya and the checkpoint weights baked in.
        image = ecr_assets.DockerImageAsset(
            self,
            "LayaImage",
            directory=str(CONTAINER_DIR),
            platform=ecr_assets.Platform.LINUX_AMD64,
            build_args={"LAYA_SUBFOLDER": config.laya_subfolder},
        )

        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for the Laya serverless endpoint",
        )
        image.repository.grant_pull(role)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                # CreateLogGroup is authorised against the bare log-group ARN,
                # CreateLogStream/PutLogEvents against "<log-group>:*".
                resources=[
                    self.format_arn(
                        service="logs",
                        resource="log-group",
                        resource_name=f"/aws/sagemaker/Endpoints/{config.endpoint_name}{suffix}",
                        arn_format=ArnFormat.COLON_RESOURCE_NAME,
                    )
                    for suffix in ("", ":*")
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],  # PutMetricData does not support resource-level permissions
                conditions={"StringEquals": {"cloudwatch:namespace": "/aws/sagemaker/Endpoints"}},
            )
        )

        model = sagemaker.CfnModel(
            self,
            "LayaModel",
            execution_role_arn=role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=image.image_uri,
                environment={
                    "LOG_LEVEL": "INFO",
                    # Only /tmp is guaranteed writable on Serverless Inference.
                    "HOME": "/tmp",
                    "XDG_CACHE_HOME": "/tmp/.cache",
                    # SageMaker replaces empty-string values with a placeholder token, so
                    # only override the subfolder baked into the image when it is set.
                    **({"LAYA_SUBFOLDER": config.laya_subfolder} if config.laya_subfolder else {}),
                },
            ),
        )
        # Make sure the role's policy is attached before SageMaker validates image access.
        model.node.add_dependency(role)

        serverless = sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
            memory_size_in_mb=config.memory_size_in_mb,
            max_concurrency=config.max_concurrency,
            provisioned_concurrency=config.provisioned_concurrency or None,
        )

        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "LayaServerlessConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.attr_model_name,
                    variant_name="AllTraffic",
                    serverless_config=serverless,
                )
            ],
        )

        endpoint = sagemaker.CfnEndpoint(
            self,
            "LayaEndpoint",
            endpoint_name=config.endpoint_name,
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )

        self.endpoint_name = endpoint.attr_endpoint_name
        self.endpoint_arn = self.format_arn(
            service="sagemaker", resource="endpoint", resource_name=config.endpoint_name.lower()
        )

        CfnOutput(self, "EndpointName", value=endpoint.attr_endpoint_name)
