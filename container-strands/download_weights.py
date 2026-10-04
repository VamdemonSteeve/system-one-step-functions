"""Build-time step: fetch the decider checkpoint and its Qwen3.5 base weights.

Only downloads files; it does not load the model, so the build host needs no extra RAM.
  - the checkpoint (LoRA adapter, head, tokenizer, config) -> a plain directory
  - the base model -> the Hugging Face cache, because strands-decider loads it by repo id
"""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

checkpoint_repo = os.environ["DECIDER_REPO"]
checkpoint_dir = Path(os.environ["DECIDER_DIR"])

snapshot_download(
    checkpoint_repo,
    local_dir=checkpoint_dir,
    ignore_patterns=["eval/*", "training/*", "history.json"],
)

config_file = checkpoint_dir / "hobson_config.json"
if not config_file.exists():
    config_file = checkpoint_dir / "strands_decider_config.json"
base_model = json.loads(config_file.read_text())["base_model"]

snapshot_download(
    base_model,
    allow_patterns=["config.json", "generation_config.json", "*.safetensors", "*.safetensors.index.json"],
)

print(f"Downloaded {checkpoint_repo} -> {checkpoint_dir} and {base_model} -> {os.environ['HF_HOME']}")
