#!/usr/bin/env python3
import os

import aws_cdk as cdk

from infra.config import LayaConfig, StrandsDeciderConfig
from infra.stacks import (
    LayaEndpointStack,
    LayaWorkflowStack,
    StrandsDeciderEndpointStack,
    StrandsDeciderImageStack,
)

app = cdk.App()
config = LayaConfig.from_context(app)

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION"),
)

endpoint_stack = LayaEndpointStack(app, "LayaEndpointStack", config=config, env=env)
workflow_stack = LayaWorkflowStack(app, "LayaWorkflowStack", config=config, env=env)
workflow_stack.add_stack_dependency(endpoint_stack)

# Second, independent serverless endpoint (not wired into the workflow). Its image is
# built and pushed by CodeBuild inside AWS, so the 4.5 GB base weights never touch this machine.
decider_config = StrandsDeciderConfig.from_context(app)
decider_image = StrandsDeciderImageStack(app, "StrandsDeciderImageStack", config=decider_config, env=env)
StrandsDeciderEndpointStack(
    app,
    "StrandsDeciderEndpointStack",
    config=decider_config,
    repository=decider_image.repository,
    image_uri=decider_image.image_uri,
    env=env,
)

cdk.Tags.of(app).add("project", "laya-serverless")

app.synth()
