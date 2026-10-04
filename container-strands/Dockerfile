# syntax=docker/dockerfile:1
# SageMaker Serverless Inference container for StrandsAgents/strands-decider-2B-hobson-v19
# (LoRA adapter + readout head on Qwen/Qwen3.5-2B-Base).
#
# Build context is the REPOSITORY ROOT (so the shared container/routing.py is reused):
#   podman build --platform linux/amd64 -f container-strands/Dockerfile -t strands-decider .
#
# Stages:
#   deps     - CPU torch + strands-decider into an isolated venv
#   weights  - downloads the checkpoint and the ~4.5 GB base weights (no model load)
#   runtime  - slim image with only the venv, the weights and the server

ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY container-strands/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
FROM deps AS weights

ARG DECIDER_REPO=StrandsAgents/strands-decider-2B-hobson-v19
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DECIDER_REPO=${DECIDER_REPO} \
    DECIDER_DIR=/opt/model/strands-decider

COPY container-strands/download_weights.py /tmp/download_weights.py
RUN python /tmp/download_weights.py \
    && rm -rf /opt/hf-cache/hub/.locks /opt/hf-cache/xet /opt/model/strands-decider/.cache \
    # Serverless runs the container under its own unprivileged UID: make everything readable.
    && chmod -R a+rX /opt/hf-cache /opt/model

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp \
    DECIDER_DIR=/opt/model/strands-decider

# No USER directive: SageMaker Serverless does not start containers that set a non-root
# USER (see container/Dockerfile). Files are root-owned and world-readable.
COPY --from=deps /opt/venv /opt/venv
COPY --from=weights /opt/hf-cache /opt/hf-cache
COPY --from=weights /opt/model /opt/model
COPY container-strands/serve.py container/routing.py /opt/program/

WORKDIR /opt/program
EXPOSE 8080
ENTRYPOINT ["python", "/opt/program/serve.py"]
