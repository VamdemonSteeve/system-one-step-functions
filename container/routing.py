"""Turn a Laya prediction into the action the caller should take.

Pure Python (no torch / laya imports) so it can be unit-tested without the model.

Request block (optional):
    "routing": {
        "question": "route",                       # which choice question decides
        "actions": {"escalate": "send_email",      # choice label -> action name
                    "auto_reply": "bedrock_reply"},
        "min_confidence": 0.7,                     # on answer_confidence, 0..1
        "fallback": "send_email"                   # low confidence or unmapped label
    }

Response block added to the prediction:
    "action": {"name": "bedrock_reply", "reason": "...", "question": "route",
               "choice": "auto_reply", "answer_confidence": 0.71, "min_confidence": 0.7}
"""

from typing import Any

DEFAULT_MIN_CONFIDENCE = 0.7


class RoutingError(ValueError):
    """The routing block is malformed or does not match the questions."""


def validate_routing(routing: Any, questions: dict) -> dict:
    """Check the routing block before inference, return it with defaults applied."""
    if not isinstance(routing, dict):
        raise RoutingError("'routing' must be an object.")

    question = routing.get("question")
    if not isinstance(question, str) or question not in questions:
        raise RoutingError("'routing.question' must name one of the questions.")
    if questions[question].get("type") != "choice":
        raise RoutingError("'routing.question' must be a 'choice' question.")

    actions = routing.get("actions")
    if (
        not isinstance(actions, dict)
        or not actions
        or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in actions.items())
    ):
        raise RoutingError("'routing.actions' must map choice labels to non-empty action names.")

    criteria = questions[question].get("criteria")
    labels = set(criteria) if isinstance(criteria, (dict, list)) else set()
    unknown = sorted(set(actions) - labels)
    if unknown:
        raise RoutingError(f"'routing.actions' has labels not in the question criteria: {unknown}")

    fallback = routing.get("fallback")
    if not isinstance(fallback, str) or not fallback:
        raise RoutingError("'routing.fallback' must be a non-empty action name.")

    min_conf = routing.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
    if isinstance(min_conf, bool) or not isinstance(min_conf, (int, float)) or not 0 <= min_conf <= 1:
        raise RoutingError("'routing.min_confidence' must be a number between 0 and 1.")

    return {
        "question": question,
        "actions": actions,
        "fallback": fallback,
        "min_confidence": float(min_conf),
    }


def decide_action(result: dict, routing: dict) -> dict:
    """Pick the action from a prediction. `routing` must come from validate_routing."""
    question = routing["question"]
    answer = result.get("answers", {}).get(question, {})
    choice = answer.get("choice")
    # answer_confidence = probability of the chosen label. Laya's `confidence` is an
    # entropy-based score on a stricter scale, so it is not used for the threshold.
    conf = answer.get("answer_confidence")
    conf = float(conf) if conf is not None else None
    min_conf = routing["min_confidence"]

    if conf is None or conf < min_conf:
        name = routing["fallback"]
        reason = f"'{choice}' below confidence threshold ({_fmt(conf)} < {min_conf})"
    elif choice not in routing["actions"]:
        name = routing["fallback"]
        reason = f"no action mapped for '{choice}'"
    else:
        name = routing["actions"][choice]
        reason = f"'{choice}' with confidence {_fmt(conf)} >= {min_conf}"

    return {
        "name": name,
        "reason": reason,
        "question": question,
        "choice": choice,
        "answer_confidence": conf,
        "min_confidence": min_conf,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
