"""SageMaker Serverless Inference endpoint serving StrandsAgents/strands-decider-2B-hobson-v19.

The image comes from StrandsDeciderImageStack (built by CodeBuild), not from a local build.
"""

from aws_cdk import ArnFormat, CfnOutput, Stack
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

from infra.config import StrandsDeciderConfig


class StrandsDeciderEndpointStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: StrandsDeciderConfig,
        repository: ecr.IRepository,
        image_uri: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for the strands-decider serverless endpoint",
        )
        repository.grant_pull(role)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
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
            "DeciderModel",
            execution_role_arn=role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=image_uri,
                environment={
                    "LOG_LEVEL": "INFO",
                    "HOME": "/tmp",
                    "XDG_CACHE_HOME": "/tmp/.cache",
                },
            ),
        )
        model.node.add_dependency(role)

        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "DeciderServerlessConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.attr_model_name,
                    variant_name="AllTraffic",
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                        memory_size_in_mb=config.memory_size_in_mb,
                        max_concurrency=config.max_concurrency,
                        provisioned_concurrency=config.provisioned_concurrency or None,
                    ),
                )
            ],
        )

        endpoint = sagemaker.CfnEndpoint(
            self,
            "DeciderEndpoint",
            endpoint_name=config.endpoint_name,
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )

        CfnOutput(self, "EndpointName", value=endpoint.attr_endpoint_name)
