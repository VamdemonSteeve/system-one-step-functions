"""Minimal SageMaker-compatible inference server for Laya.

SageMaker contract:
  GET  /ping         -> 200 when the model is loaded
  POST /invocations  -> JSON in, JSON out, listening on port 8080

Request body:
  {"state": <str | object>,
   "questions": {<name>: {"type": ..., "instructions": ..., ...}},
   "routing": {...}}   # optional, see routing.py

Response body: the dict returned by laya's Agent.predict (answers, usage, ...), plus a
top-level "action" object (the action to take) when "routing" is supplied.
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager

import laya
import torch
import uvicorn
from fastapi import FastAPI, Request, Response

from routing import RoutingError, decide_action, validate_routing

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("laya-serve")

MODEL_REPO = os.environ.get("LAYA_REPO", "convaiinnovations/laya")
SUBFOLDER = os.environ.get("LAYA_SUBFOLDER") or None
THREADS = os.environ.get("LAYA_THREADS")

_agent = None
# Serialise forward passes; SageMaker scales out with more containers, not threads.
_lock = threading.Lock()


def _to_builtin(obj):
    """json.dumps fallback for numpy / torch scalars and arrays."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _json(payload: dict, status: int = 200) -> Response:
    return Response(
        content=json.dumps(payload, default=_to_builtin),
        status_code=status,
        media_type="application/json",
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent
    if THREADS:
        torch.set_num_threads(int(THREADS))
    kwargs = {"subfolder": SUBFOLDER} if SUBFOLDER else {}
    log.info("Loading %s (subfolder=%s)", MODEL_REPO, SUBFOLDER)
    _agent = laya.load(MODEL_REPO, **kwargs)
    log.info("Model loaded")
    yield
    _agent = None


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping() -> Response:
    return Response(status_code=200 if _agent is not None else 503)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _json({"error": "Request body must be valid JSON."}, 400)

    if not isinstance(body, dict):
        return _json({"error": "Request body must be a JSON object."}, 400)

    state = body.get("state")
    questions = body.get("questions")
    if state is None or not isinstance(questions, dict) or not questions:
        return _json({"error": "Body needs 'state' and a non-empty 'questions' object."}, 400)

    routing = None
    if "routing" in body:
        try:
            routing = validate_routing(body["routing"], questions)
        except RoutingError as exc:
            return _json({"error": str(exc)}, 400)

    try:
        with _lock:
            result = _agent.predict(state, questions)
    except (ValueError, TypeError, KeyError) as exc:
        # laya raises on malformed question schemas; surface that as a client error.
        return _json({"error": f"Invalid request: {exc}"}, 400)
    except Exception:  # noqa: BLE001
        log.exception("Inference failed")
        return _json({"error": "Inference failed."}, 500)

    if routing is not None:
        result["action"] = decide_action(result, routing)

    return _json(result)


if __name__ == "__main__":
    # SageMaker passes "serve" as argv[1]; it is ignored here.
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1, log_level="info")
