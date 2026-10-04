"""CloudFormation custom resource: run a CodeBuild build and wait for it to finish.

Used with the CDK Provider framework:
  on_event    - Create/Update: start a build with IMAGE_TAG; Delete: nothing to do
  is_complete - polled until the build succeeds (done) or fails (raises -> stack rollback)

The physical resource id is the image tag, so a new tag (new source) means a new build,
and an unchanged tag never triggers one.
"""

import boto3

codebuild = boto3.client("codebuild")

_FAILED = {"FAILED", "FAULT", "STOPPED", "TIMED_OUT"}


def on_event(event, _context):
    props = event["ResourceProperties"]
    tag = props["ImageTag"]

    if event["RequestType"] == "Delete":
        # Images are cleaned up with the repository; there is no build to undo.
        return {"PhysicalResourceId": event.get("PhysicalResourceId", tag)}

    build = codebuild.start_build(
        projectName=props["ProjectName"],
        environmentVariablesOverride=[{"name": "IMAGE_TAG", "value": tag, "type": "PLAINTEXT"}],
    )["build"]
    print(f"Started {build['id']} for tag {tag}")
    return {"PhysicalResourceId": tag, "Data": {"BuildId": build["id"], "ImageTag": tag}}


def is_complete(event, _context):
    if event["RequestType"] == "Delete":
        return {"IsComplete": True}

    build_id = event["Data"]["BuildId"]
    build = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
    status = build["buildStatus"]
    print(f"{build_id}: {status} (phase {build.get('currentPhase')})")

    if status == "SUCCEEDED":
        return {"IsComplete": True, "Data": event["Data"]}
    if status in _FAILED:
        logs = build.get("logs", {}).get("deepLink", "see the CodeBuild console")
        raise RuntimeError(f"Image build {build_id} ended with {status}. Logs: {logs}")
    return {"IsComplete": False}
