# syntax=docker/dockerfile:1
# SageMaker Serverless Inference container for convaiinnovations/laya.
# Serverless endpoints are CPU-only and x86_64, so build with --platform linux/amd64
# (the CDK DockerImageAsset does this for you).
#
# Stages:
#   deps     - installs CPU torch + laya into an isolated venv
#   weights  - uses that venv to download the checkpoint into /opt/hf-cache
#   runtime  - clean slim image with only the venv, the weights and serve.py

ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim

# ---------------------------------------------------------------------------
# Stage 1: Python dependencies
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CPU-only PyTorch wheel keeps the image small (no CUDA libraries).
# Installed in its own layer so changes to requirements.txt don't re-download torch.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: model weights
# ---------------------------------------------------------------------------
FROM deps AS weights

# Which checkpoint to bake in: "" = English root, "multilingual", or "typed-decisions".
ARG LAYA_REPO=convaiinnovations/laya
ARG LAYA_SUBFOLDER=""

ENV USE_TF=0 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LAYA_REPO=${LAYA_REPO} \
    LAYA_SUBFOLDER=${LAYA_SUBFOLDER}

COPY download_weights.py /tmp/download_weights.py
RUN python /tmp/download_weights.py \
    # Drop hub lock files and any partial downloads; only the snapshot is needed.
    && rm -rf /opt/hf-cache/hub/.locks /opt/hf-cache/xet \
    # Serverless Inference runs the container under its own unprivileged UID, so every
    # file must be world-readable (the Hub writes some cache files as 0600).
    && chmod -R a+rX /opt/hf-cache

# ---------------------------------------------------------------------------
# Stage 3: runtime
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG LAYA_REPO=convaiinnovations/laya
ARG LAYA_SUBFOLDER=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    # transformers probes for TensorFlow at import; laya docs recommend disabling it.
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf-cache \
    # Never reach out to the Hub at runtime; weights are baked in.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LAYA_REPO=${LAYA_REPO} \
    LAYA_SUBFOLDER=${LAYA_SUBFOLDER}

# No USER directive: SageMaker Serverless Inference fails to start containers that set a
# non-root USER (verified: even a trivial server in this image never becomes healthy).
# Serverless already runs the process under an unprivileged UID in an isolated microVM.
# Everything below is root-owned and world-readable; nothing is written outside /tmp.
COPY --from=deps /opt/venv /opt/venv
COPY --from=weights /opt/hf-cache /opt/hf-cache
COPY serve.py routing.py /opt/program/

WORKDIR /opt/program

EXPOSE 8080
# SageMaker starts the container with the argument "serve".
ENTRYPOINT ["python", "/opt/program/serve.py"]
