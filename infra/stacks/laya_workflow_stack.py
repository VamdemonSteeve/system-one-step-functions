"""Step Functions workflow: Laya decision -> Choice -> SNS email | Bedrock reply."""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_stepfunctions as sfn
from constructs import Construct

from infra.config import LayaConfig

ASL_FILE = Path(__file__).resolve().parents[2] / "statemachine" / "laya_router.asl.json"

# Cross-Region inference profile prefixes (e.g. "us.amazon.nova-lite-v1:0").
_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "us-gov.", "ca.", "jp.", "au.", "global.")


class LayaWorkflowStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, config: LayaConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- SNS topic for escalations (email) -------------------------------------------
        topic = sns.Topic(
            self,
            "EscalationTopic",
            display_name="Laya escalations",
            master_key=kms.Alias.from_alias_name(self, "SnsManagedKey", "alias/aws/sns"),
            enforce_ssl=True,
        )
        if config.notification_email:
            # The recipient must confirm the subscription email before messages arrive.
            topic.add_subscription(subs.EmailSubscription(config.notification_email))

        # --- Encrypted execution logs ---------------------------------------------------
        log_key = kms.Key(
            self,
            "WorkflowLogKey",
            enable_key_rotation=True,
            description="Encrypts Laya workflow execution logs",
            removal_policy=RemovalPolicy.DESTROY,
        )
        log_key.grant_encrypt_decrypt(iam.ServicePrincipal(f"logs.{self.region}.amazonaws.com"))

        log_group = logs.LogGroup(
            self,
            "WorkflowLogs",
            log_group_name=f"/aws/vendedlogs/states/{construct_id}",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=log_key,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --- State machine --------------------------------------------------------------
        state_machine = sfn.StateMachine(
            self,
            "LayaRouter",
            state_machine_type=sfn.StateMachineType.STANDARD,
            definition_body=sfn.DefinitionBody.from_string(ASL_FILE.read_text()),
            definition_substitutions={
                "EndpointName": config.endpoint_name,
                "TopicArn": topic.topic_arn,
                "BedrockModelId": config.bedrock_model_id,
                "ConfidenceThreshold": str(config.confidence_threshold),
            },
            timeout=Duration.minutes(10),
            tracing_enabled=True,
            logs=sfn.LogOptions(
                destination=log_group,
                level=sfn.LogLevel.ALL,
                include_execution_data=True,
            ),
        )

        # --- Least-privilege permissions for the execution role ------------------------
        state_machine.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint"],
                resources=[
                    self.format_arn(
                        service="sagemaker",
                        resource="endpoint",
                        resource_name=config.endpoint_name.lower(),
                    )
                ],
            )
        )
        topic.grant_publish(state_machine)
        state_machine.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=self._bedrock_resources(config.bedrock_model_id),
            )
        )

        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        CfnOutput(self, "EscalationTopicArn", value=topic.topic_arn)

    def _bedrock_resources(self, model_id: str) -> list[str]:
        """Model ARNs for either a plain foundation model or a cross-Region inference profile."""
        prefix = next((p for p in _PROFILE_PREFIXES if model_id.startswith(p)), None)
        if prefix is None:
            return [f"arn:{self.partition}:bedrock:{self.region}::foundation-model/{model_id}"]

        base_model = model_id[len(prefix):]
        return [
            f"arn:{self.partition}:bedrock:{self.region}:{self.account}:inference-profile/{model_id}",
            # A profile routes to the base model in any Region it covers.
            f"arn:{self.partition}:bedrock:*::foundation-model/{base_model}",
        ]
