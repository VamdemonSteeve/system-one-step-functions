#!/usr/bin/env bash
# Run only the DraftReplyWithBedrock state against real Bedrock (TestState API),
# using the deployed state machine's IAM role. Nothing is deployed.
#
#   AWS_PROFILE=deloitte AWS_REGION=eu-central-1 scripts/test_bedrock_step.sh "How do I reset my password?"
#   scripts/test_bedrock_step.sh '{"subject":"Book","body":"What is Sapiens about?"}'
set -euo pipefail

cd "$(dirname "$0")/.."
MESSAGE_ARG="${1:?usage: $0 <message text or JSON object>}"
MODEL_ID="${BEDROCK_MODEL_ID:-$(python3 -c 'import json;print(json.load(open("cdk.json"))["context"]["laya"]["bedrockModelId"])')}"

SM_ARN=$(aws cloudformation describe-stacks --stack-name LayaWorkflowStack \
  --query "Stacks[0].Outputs[?OutputKey=='StateMachineArn'].OutputValue" --output text)
ROLE_ARN=$(aws stepfunctions describe-state-machine --state-machine-arn "$SM_ARN" --query roleArn --output text)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python3 - "$MODEL_ID" "$MESSAGE_ARG" "$TMP" <<'EOF'
import json, sys
model_id, message_arg, tmp = sys.argv[1:4]
state = json.load(open("statemachine/laya_router.asl.json"))["States"]["DraftReplyWithBedrock"]
state["QueryLanguage"] = "JSONata"
state.pop("Retry", None)
open(f"{tmp}/state.json", "w").write(json.dumps(state).replace("${BedrockModelId}", model_id))
try:
    message = json.loads(message_arg)          # JSON object input, e.g. {"subject":..,"body":..}
except json.JSONDecodeError:
    message = message_arg                      # plain text
variables = {"message": message, "action": {"name": "bedrock_reply", "reason": "test_bedrock_step.sh"}}
open(f"{tmp}/vars.json", "w").write(json.dumps(variables))
EOF

aws stepfunctions test-state \
  --definition "file://$TMP/state.json" \
  --role-arn "$ROLE_ARN" \
  --input '{}' \
  --variables "file://$TMP/vars.json" \
  --query '[status, output, error, cause]' --output json \
| python3 -c '
import json, sys
status, output, error, cause = json.load(sys.stdin)
print("status:", status)
if output:
    print(json.loads(output)["reply"])
else:
    print("error:", error, "\ncause:", cause)
'
