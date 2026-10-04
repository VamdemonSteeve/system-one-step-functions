"""Builds the strands-decider container image in AWS CodeBuild and publishes it to ECR.

Nothing large leaves your machine: only the build context (a few KB of Dockerfile and
Python) is uploaded as an S3 asset. CodeBuild downloads torch and the ~4.5 GB base
weights and pushes the image from inside AWS.

`cdk deploy` blocks until the build has finished (custom resource), so stacks that use
`image_uri` only deploy once the image exists. The tag is the source asset hash, so a
rebuild happens exactly when the build context changes.
"""

from pathlib import Path

from aws_cdk import CfnOutput, CustomResource, Duration, IgnoreMode, RemovalPolicy, Stack
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3_assets as s3_assets
from aws_cdk import custom_resources as cr
from constructs import Construct

from infra.config import StrandsDeciderConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIGGER_CODE = Path(__file__).resolve().parents[1] / "lambda" / "codebuild_trigger"

BUILD_TIMEOUT = Duration.minutes(60)

# The Dockerfile expects the repository root as its context.
SOURCE_EXCLUDE = ["*", "!container-strands/**", "!container/routing.py", "**/__pycache__"]

BUILDSPEC = {
    "version": "0.2",
    "env": {"shell": "bash"},
    "phases": {
        "pre_build": {
            "commands": [
                "aws ecr get-login-password --region $AWS_DEFAULT_REGION"
                " | docker login --username AWS --password-stdin ${REPO_URI%%/*}",
            ]
        },
        "build": {
            "commands": [
                "docker build --platform linux/amd64 -f container-strands/Dockerfile"
                " --build-arg DECIDER_REPO=$DECIDER_REPO -t $REPO_URI:$IMAGE_TAG .",
                "docker images $REPO_URI:$IMAGE_TAG",
            ]
        },
        "post_build": {
            "commands": [
                # post_build runs even after a failed build; don't push a missing image.
                'if [ "$CODEBUILD_BUILD_SUCCEEDING" != "1" ]; then exit 1; fi',
                "docker push $REPO_URI:$IMAGE_TAG",
                # SageMaker rejects OCI manifests; fail loudly if the push produced one.
                "MEDIA=$(aws ecr batch-get-image --repository-name ${REPO_URI#*/}"
                " --image-ids imageTag=$IMAGE_TAG --query 'images[0].imageManifestMediaType' --output text)",
                'echo "manifest: $MEDIA"',
                'test "$MEDIA" = "application/vnd.docker.distribution.manifest.v2+json"',
            ]
        },
    },
}


class StrandsDeciderImageStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, config: StrandsDeciderConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        repository = ecr.Repository(
            self,
            "DeciderRepository",
            image_scan_on_push=True,
            encryption=ecr.RepositoryEncryption.AES_256,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=3, description="keep the last 3 builds")],
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
        )

        source = s3_assets.Asset(
            self,
            "DeciderBuildContext",
            path=str(REPO_ROOT),
            exclude=SOURCE_EXCLUDE,
            ignore_mode=IgnoreMode.DOCKER,
        )
        image_tag = source.asset_hash

        build_logs = logs.LogGroup(
            self,
            "BuildLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        project = codebuild.Project(
            self,
            "DeciderImageBuild",
            description="Builds and pushes the strands-decider SageMaker image",
            source=codebuild.Source.s3(bucket=source.bucket, path=source.s3_object_key),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                # LARGE: 8 vCPU / 15 GB RAM / 128 GB disk; the image is ~7 GB unpacked.
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,  # required to run the Docker daemon
            ),
            environment_variables={
                "REPO_URI": codebuild.BuildEnvironmentVariable(value=repository.repository_uri),
                "DECIDER_REPO": codebuild.BuildEnvironmentVariable(value=config.checkpoint_repo),
                "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value=image_tag),
            },
            build_spec=codebuild.BuildSpec.from_object(BUILDSPEC),
            timeout=BUILD_TIMEOUT,
            logging=codebuild.LoggingOptions(
                cloud_watch=codebuild.CloudWatchLoggingOptions(log_group=build_logs)
            ),
        )
        source.grant_read(project)
        repository.grant_pull_push(project)

        # --- Run the build during deployment and wait for it -----------------------------
        trigger_role = iam.Role(
            self,
            "BuildTriggerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ],
        )
        trigger_role.add_to_policy(
            iam.PolicyStatement(
                actions=["codebuild:StartBuild", "codebuild:BatchGetBuilds"],
                resources=[project.project_arn],
            )
        )

        def trigger_fn(name: str, handler: str) -> lambda_.Function:
            return lambda_.Function(
                self,
                name,
                runtime=lambda_.Runtime.PYTHON_3_12,
                handler=handler,
                code=lambda_.Code.from_asset(str(TRIGGER_CODE)),
                timeout=Duration.seconds(30),
                role=trigger_role,
                log_group=logs.LogGroup(
                    self,
                    f"{name}Logs",
                    retention=logs.RetentionDays.ONE_WEEK,
                    removal_policy=RemovalPolicy.DESTROY,
                ),
            )

        provider = cr.Provider(
            self,
            "BuildProvider",
            on_event_handler=trigger_fn("BuildStartFn", "index.on_event"),
            is_complete_handler=trigger_fn("BuildWaitFn", "index.is_complete"),
            query_interval=Duration.seconds(30),
            total_timeout=BUILD_TIMEOUT,
        )

        build = CustomResource(
            self,
            "ImageBuild",
            service_token=provider.service_token,
            properties={"ProjectName": project.project_name, "ImageTag": image_tag},
        )

        # Consumers reference the tag through the custom resource, which makes them wait
        # for the build instead of racing it.
        self.repository = repository
        self.image_uri = f"{repository.repository_uri}:{build.get_att_string('ImageTag')}"

        CfnOutput(self, "ImageUri", value=self.image_uri)
        CfnOutput(self, "BuildProject", value=project.project_name)
