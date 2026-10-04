# Laya on SageMaker Serverless + Step Functions router

Deploys [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (a calibrated, non-generative decision model) to a SageMaker Serverless Inference endpoint, and a Step Functions workflow that uses its decision to either email a human via SNS or draft a reply with Bedrock.

```
            ┌────────────────────┐  action.name =      ┌──────────────────────┐
 input ──▶  │ ClassifyWithLaya   │ ─ send_email ─────▶ │ EscalateViaSnsEmail  │
 {message}  │ (SageMaker         │   (or anything else)└──────────────────────┘
            │  Serverless)       │                     ┌──────────────────────┐
            └────────┬───────────┘ ─ bedrock_reply ──▶ │ DraftReplyWithBedrock│
                     │  RouteOnAction (Choice)         │ (Converse API)       │
                                                       └──────────────────────┘
```

The endpoint decides the action. The request carries an optional `routing` block, and the response then includes a top-level `action` object that the Choice state branches on:

```json
"routing": {"question": "route",
            "actions": {"auto_reply": "bedrock_reply", "escalate": "send_email"},
            "min_confidence": 0.7, "fallback": "send_email"}

"action": {"name": "send_email",
           "reason": "'auto_reply' below confidence threshold (0.6993 < 0.7)",
           "question": "route", "choice": "auto_reply",
           "answer_confidence": 0.6993, "min_confidence": 0.7}
```

The threshold is compared with `answer_confidence` (probability of the chosen label). Low confidence or an unmapped label gives `fallback`. Without `routing`, the response is the plain Laya prediction. The logic lives in `container/routing.py`.

## Layout

```
app.py                         CDK app (four stacks)
cdk.json                       CDK config; "laya" and "strandsDecider" context blocks
infra/
  config.py                    typed config + validation of SageMaker Serverless limits
  stacks/laya_endpoint_stack.py   ECR image asset, IAM role, Model, serverless EndpointConfig, Endpoint
  stacks/laya_workflow_stack.py   SNS topic, KMS-encrypted logs, state machine, least-privilege role
container/
  Dockerfile                   CPU torch + laya, weights baked in at build time
  serve.py                     /ping + /invocations server on :8080 (SageMaker contract)
  routing.py                   maps a prediction to the action to take (no model imports)
  download_weights.py          build-time checkpoint download
container-strands/             second endpoint: strands-decider-2B (build context = repo root)
  Dockerfile, serve.py, download_weights.py, requirements.txt
infra/stacks/strands_decider_image_stack.py      ECR repo + CodeBuild project + build-and-wait custom resource
infra/stacks/strands_decider_endpoint_stack.py   its serverless endpoint
infra/lambda/codebuild_trigger/index.py          starts the build and polls until it finishes
statemachine/
  laya_router.asl.json         JSONata ASL definition (${...} placeholders filled by CDK)
events/                        sample workflow inputs
```

## Second endpoint: strands-decider

`StrandsDeciderEndpointStack` deploys [StrandsAgents/strands-decider-2B-hobson-v19](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) (a LoRA adapter and readout head on Qwen/Qwen3.5-2B-Base) to its own serverless endpoint, `strands-decider-serverless`. It is independent of the workflow.

- Same request format as the Laya endpoint (`state`, `questions`, optional `routing`), so the state machine could call either. Choice answers get an `answer_confidence` field (probability of the chosen label) so `routing.min_confidence` means the same thing on both endpoints. The response also includes `latency_ms`.
- The 2B base runs in **bf16 on CPU**. The library's default upcasts to fp32 on CPU (~7 GiB of weights), which cannot fit in Serverless's 6 GB maximum. Set `DECIDER_CPU_FP32=1` on a real-time instance with more memory if you want the faster fp32 path.
- The image is about 6–7 GB (the base weights are 4.5 GB), under the 10 GB Serverless limit.
- The image is **built in AWS by CodeBuild**, not locally. `StrandsDeciderImageStack` uploads only the build context (a few KB from `container-strands/` plus `container/routing.py`) as an S3 asset. A CodeBuild project (LARGE, privileged Docker) builds the image, downloads the weights from Hugging Face and pushes it to a dedicated ECR repository. A custom resource starts the build during `cdk deploy` and waits for it (up to 60 min), so the endpoint stack only deploys once the image exists. The image tag is the build-context hash: a rebuild happens only when those files change. The build fails if ECR reports an OCI manifest, which SageMaker rejects.

```bash
cdk deploy StrandsDeciderImageStack StrandsDeciderEndpointStack   # no local Docker/Podman needed
```

Build logs are in CloudWatch, in the CodeBuild project's log group (see the `BuildProject` stack output).

Config lives in `cdk.json` → `context.strandsDecider` (`endpointName`, `memorySizeInMb`, `maxConcurrency`, `provisionedConcurrency`, `checkpointRepo`).

## Configuration (`cdk.json` → `context.laya`)

| key | default | notes |
|---|---|---|
| `endpointName` | `laya-serverless` | |
| `memorySizeInMb` | `6144` | Serverless max. The English checkpoint is ~808 MB of weights plus torch. |
| `maxConcurrency` | `5` | 1–200 |
| `provisionedConcurrency` | `0` | Set > 0 to keep warm containers and avoid cold starts (billed while idle). |
| `layaSubfolder` | `""` | `""` = English root, `multilingual`, or `typed-decisions` |
| `bedrockModelId` | `eu.anthropic.claude-haiku-4-5-20251001-v1:0` | Model or cross-Region inference profile ID used to draft replies. Match the prefix (`us.`, `eu.`, `apac.`) to your Region. Nova Lite/Pro also work but tended to guess company policies (e.g. "30-day returns") despite the prompt. |
| `confidenceThreshold` | `0.5` | Sent as `routing.min_confidence`. If `answer_confidence` is below it, the endpoint returns the fallback action (`send_email`). On a two-option question, 0.5 means "follow Laya's top choice"; the base checkpoint rarely exceeds 0.7, so higher values escalate almost everything. |
| `notificationEmail` | `""` | If set, an email subscription is created (confirm it from your inbox). |

Override on the command line, e.g. `cdk deploy --all -c laya='{"notificationEmail":"you@example.com"}'`. Note that `-c laya=...` replaces the whole block, so unspecified keys fall back to the defaults in `infra/config.py`.

## Deploy

Requires Podman (or Docker), Python 3.10+, and AWS CDK CLI **>= 2.1143.0** (`aws-cdk-lib` 2.271 emits a cloud-assembly schema older CLIs can't read; `npm i -g aws-cdk@latest`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

podman machine start
export CDK_DOCKER=podman      # make CDK build/push image assets with Podman
export BUILDAH_FORMAT=docker  # SageMaker rejects OCI manifests; Podman's default is OCI

cdk bootstrap            # once per account/Region
cdk deploy --all
```

The first build downloads torch and the checkpoint into a `linux/amd64` image; on Apple Silicon this runs under emulation and takes a while.

## Run locally

SageMaker needs `linux/amd64`, which runs under emulation on Apple Silicon (over a minute per request). For local work, build a native arm64 image from the same Dockerfile:

```bash
podman build --platform linux/arm64 -t laya-sm:local container/
podman run -d --name laya-local -p 8080:8080 laya-sm:local serve   # ready in ~10 s
curl -i localhost:8080/ping

curl -s localhost:8080/invocations -H 'content-type: application/json' -d '{
  "state": "Hi, we were billed twice for March. Please refund the duplicate.",
  "questions": {"route": {"type": "choice",
    "instructions": "Can this message be answered with a standard written reply, or does a human need to handle it?",
    "criteria": {"auto_reply": "routine question", "escalate": "complaint, refund, billing dispute, urgent"}}}
}'

podman rm -f laya-local   # stop
```

Warm requests take roughly 0.4–0.9 s on a 2-vCPU Podman machine.

## Try it

```bash
SM_ARN=$(aws cloudformation describe-stacks --stack-name LayaWorkflowStack \
  --query "Stacks[0].Outputs[?OutputKey=='StateMachineArn'].OutputValue" --output text)

aws stepfunctions start-execution --state-machine-arn "$SM_ARN" --input file://events/escalate.json
aws stepfunctions start-execution --state-machine-arn "$SM_ARN" --input file://events/auto_reply.json
```

Try only the Bedrock reply step (real Bedrock call via the TestState API, using the deployed state machine role; nothing is deployed):

```bash
scripts/test_bedrock_step.sh "Can you explain the ending of The Great Gatsby?"
scripts/test_bedrock_step.sh '{"subject":"Returns","body":"How long do I have to return a book?"}'
```

Call the endpoint directly:

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name laya-serverless \
  --content-type application/json --cli-binary-format raw-in-base64-out \
  --body '{"state":"I was charged twice, refund me","questions":{"route":{"type":"choice","instructions":"Who handles this?","criteria":{"auto_reply":"routine","escalate":"needs a human"}}}}' \
  /dev/stdout
```

## Things to know

- **Cold starts.** Serverless containers scale to zero. The first call after idle loads a ~3 GB image and the model, which can take tens of seconds; the workflow retries the SageMaker task with backoff. Use `provisionedConcurrency` if latency matters.
- **CPU only.** Serverless has no GPU; expect a few hundred ms per call once warm (the model card quotes 193–464 ms on CPU).
- **Zero-shot accuracy.** The model card is explicit that base checkpoints are a starting point for fine-tuning, and ship over-confident until you fit temperatures on your data. Validate the `route` question in `statemachine/laya_router.asl.json` against real messages before relying on it. It uses a `choice` with semantic keys rather than `noul`, which the card flags as label-sensitive on the English checkpoint.
- **Email content.** The SNS email carries only the execution ID and decision, not the customer message. The message stays in the (KMS-encrypted) execution logs and history.
- **Endpoint access.** The endpoint is reachable only via IAM-signed `InvokeEndpoint` calls; the state machine role is scoped to this endpoint, topic and Bedrock model.
