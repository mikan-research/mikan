"""Semantic frames for structured-output tasks: parse, canonicalise, score, reward.

A frame is ``{"intent": str, "slots": [{"type": str, "value": str}, ...]}``. Slots are a
list because a type can repeat ("from [date : monday] to [date : friday]"). Everything a
program can check about an output lives here, so evaluation and the GRPO reward use the
same definitions:

* valid: the output is exactly one JSON object with that shape, the intent and every slot
  type are in the schema;
* grounded: every slot value occurs in the input text;
* exact match: same intent and the same multiset of (type, value) pairs.

Stdlib only (it runs inside Kaggle and Modal containers).
"""

from __future__ import annotations

import json
import re
from collections import Counter

_ANNOT = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")
_WS = re.compile(r"\s+")


def norm(value: str) -> str:
    """Slot values compare case- and whitespace-insensitively."""
    return _WS.sub(" ", value).strip().lower()


def parse_annotation(annot: str) -> list[dict]:
    """MASSIVE ``annot_utt`` ("wake me at [time : nine am]") -> slot list."""
    return [{"type": t, "value": v} for t, v in _ANNOT.findall(annot)]


def parse(raw, intents: set[str] | None = None, slot_types: set[str] | None = None) -> dict | None:
    """Return the frame if ``raw`` is a valid frame, else None. Only surrounding whitespace is forgiven."""
    if raw is None:
        return None
    try:
        obj = json.loads(str(raw).strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"intent", "slots"}:
        return None
    intent, slots = obj["intent"], obj["slots"]
    if not isinstance(intent, str) or not isinstance(slots, list):
        return None
    if intents is not None and intent not in intents:
        return None
    out = []
    for s in slots:
        if not isinstance(s, dict) or set(s) != {"type", "value"}:
            return None
        if not isinstance(s["type"], str) or not isinstance(s["value"], str) or not s["value"].strip():
            return None
        if slot_types is not None and s["type"] not in slot_types:
            return None
        out.append({"type": s["type"], "value": s["value"]})
    return {"intent": intent, "slots": out}


def _pairs(frame: dict) -> Counter:
    return Counter((s["type"], norm(s["value"])) for s in frame["slots"])


def canonical(frame: dict) -> str:
    pairs = sorted(_pairs(frame).elements())
    return json.dumps({"intent": frame["intent"], "slots": pairs}, separators=(",", ":"))


def to_json(frame: dict) -> str:
    """The exact string a model is trained to emit."""
    return json.dumps({"intent": frame["intent"], "slots": frame["slots"]}, ensure_ascii=False)


def exact_match(pred: dict | None, ref: dict) -> bool:
    return pred is not None and pred["intent"] == ref["intent"] and _pairs(pred) == _pairs(ref)


def slot_counts(pred: dict | None, ref: dict) -> tuple[int, int, int]:
    """(true positives, predicted, reference) over the (type, value) multisets."""
    r = _pairs(ref)
    if pred is None:
        return 0, 0, sum(r.values())
    p = _pairs(pred)
    return sum((p & r).values()), sum(p.values()), sum(r.values())


def slot_f1(pred: dict | None, ref: dict) -> float:
    tp, n_p, n_r = slot_counts(pred, ref)
    if n_p == 0 and n_r == 0:
        return 1.0 if pred is not None else 0.0
    return 2 * tp / (n_p + n_r)


def grounded(frame: dict, text: str) -> float:
    """Fraction of slot values that occur in the text (1.0 when there are no slots)."""
    if not frame["slots"]:
        return 1.0
    t = norm(text)
    return sum(norm(s["value"]) in t for s in frame["slots"]) / len(frame["slots"])


def reward(raw, ref: dict, text: str, intents: set[str], slot_types: set[str]) -> float:
    """GRPO reward in [0, 1] against a reference frame (the teacher's output, never golden).

    0 for an invalid output; otherwise 0.1 for validity, 0.3 for the intent, 0.6 x slot F1,
    with the slot part scaled by groundedness so invented spans are never rewarded.
    """
    pred = parse(raw, intents, slot_types)
    if pred is None:
        return 0.0
    intent_ok = float(pred["intent"] == ref["intent"])
    return 0.1 + 0.3 * intent_ok + 0.6 * slot_f1(pred, ref) * grounded(pred, text)


def json_schema(intents: list[str], slot_types: list[str]) -> dict:
    """JSON schema for constrained decoding (vLLM structured outputs)."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(intents)},
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": list(slot_types)},
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["type", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "slots"],
        "additionalProperties": False,
    }


def summary(golds: list[dict], preds: list[dict | None], texts: list[str]) -> dict:
    """Frame-level metrics for one system over a split."""
    n = len(golds)
    tp = n_p = n_r = 0
    for p, g in zip(preds, golds):
        a, b, c = slot_counts(p, g)
        tp, n_p, n_r = tp + a, n_p + b, n_r + c
    valid = [p for p in preds if p is not None]
    return {
        "exact_match": sum(exact_match(p, g) for p, g in zip(preds, golds)) / n,
        "intent_accuracy": sum(p is not None and p["intent"] == g["intent"] for p, g in zip(preds, golds)) / n,
        "slot_precision": tp / n_p if n_p else 0.0,
        "slot_recall": tp / n_r if n_r else 0.0,
        "slot_f1": 2 * tp / (n_p + n_r) if n_p + n_r else 0.0,
        "validity_rate": len(valid) / n,
        "grounded_rate": (sum(grounded(p, t) for p, t in zip(preds, texts) if p is not None) / len(valid)) if valid else 0.0,
    }
