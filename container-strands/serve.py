"""SageMaker-compatible inference server for StrandsAgents/strands-decider-2B-hobson-v19.

SageMaker contract:
  GET  /ping         -> 200 when the model is loaded
  POST /invocations  -> JSON in, JSON out, listening on port 8080

Request body (same shape as the Laya endpoint, so callers can switch between them):
  {"state": <str | object>,
   "questions": {<name>: {"type": "choice" | "score" | "noul", "instructions": ..., ...}},
   "routing": {...}}   # optional, see routing.py

Response body: the strands-decider SystemOne response (model, answers, usage), with
`answer_confidence` (probability of the chosen label) added to every choice answer,
`latency_ms`, and a top-level "action" object when "routing" is supplied.
"""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import ValidationError
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.schema import SystemOneRequest

from routing import RoutingError, decide_action, validate_routing

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("strands-decider-serve")

CHECKPOINT_DIR = os.environ.get("DECIDER_DIR", "/opt/model/strands-decider")
MODEL_NAME = os.environ.get("DECIDER_MODEL_NAME", "strands-decider-2B-hobson-v19")
THREADS = os.environ.get("DECIDER_THREADS")
# The engine upcasts the torso to fp32 on CPU (faster per token, but ~7 GiB of weights).
# That cannot fit in a 6 GB Serverless endpoint, so it stays in bf16 unless asked.
CPU_FP32 = os.environ.get("DECIDER_CPU_FP32", "0") == "1"

_engine: SystemOneEngine | None = None
# Serialise forward passes; SageMaker scales out with more containers, not threads.
_lock = threading.Lock()


def _json(payload: dict, status: int = 200) -> Response:
    return Response(content=json.dumps(payload), status_code=status, media_type="application/json")


@torch.inference_mode()
def _load_engine() -> SystemOneEngine:
    if not CPU_FP32:
        SystemOneEngine._upcast_torso_for_cpu = lambda self: None  # keep bf16 weights
    model = StrandsDeciderModel.load(CHECKPOINT_DIR)
    return SystemOneEngine(model, EngineConfig(device="cpu", model_name=MODEL_NAME))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    if THREADS:
        torch.set_num_threads(int(THREADS))
    started = time.perf_counter()
    log.info("Loading %s (cpu, %s)", CHECKPOINT_DIR, "fp32" if CPU_FP32 else "bf16")
    _engine = _load_engine()
    log.info("Model loaded in %.1f s", time.perf_counter() - started)
    yield
    _engine = None


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping() -> Response:
    return Response(status_code=200 if _engine is not None else 503)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _json({"error": "Request body must be valid JSON."}, 400)
    if not isinstance(body, dict):
        return _json({"error": "Request body must be a JSON object."}, 400)

    try:
        req = SystemOneRequest.model_validate(
            {"state": body.get("state"), "questions": body.get("questions")}
        )
    except ValidationError as exc:
        return _json({"error": "Invalid request.", "details": exc.errors(include_url=False, include_context=False)}, 400)

    routing = None
    if "routing" in body:
        try:
            routing = validate_routing(body["routing"], body["questions"])
        except RoutingError as exc:
            return _json({"error": str(exc)}, 400)

    started = time.perf_counter()
    try:
        with _lock, torch.inference_mode():
            response = _engine.evaluate(req)
    except ValueError as exc:
        # Too many options for the head's slots, prompt truncated through options, etc.
        return _json({"error": f"Invalid request: {exc}"}, 400)
    except Exception:  # noqa: BLE001
        log.exception("Inference failed")
        return _json({"error": "Inference failed."}, 500)

    result = response.model_dump()
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    for answer in result["answers"].values():
        if answer.get("type") == "choice":
            # strands-decider's `confidence` is a normalised score; routing thresholds on
            # the probability of the chosen label, as for the Laya endpoint.
            answer["answer_confidence"] = answer["probabilities"][answer["choice"]]

    if routing is not None:
        result["action"] = decide_action(result, routing)
    return _json(result)


if __name__ == "__main__":
    # SageMaker passes "serve" as argv[1]; it is ignored here.
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1, log_level="info")
