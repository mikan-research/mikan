"""Training targets and subsets for SFT. Standard library only, so it is unit-testable anywhere.

A student is trained on a short prompt (instruction + message, no label list and no
examples). It must learn the label set itself; serving later constrains output to the
77 labels, and the short prompt is what keeps inference cheap.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

STUDENT_SYSTEM = (
    "Classify the online-banking customer message into exactly one intent. "
    "Reply with the intent label only."
)

TARGET_SOURCES = ("teacher", "gold")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def student_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": text}]


def attach_targets(rows: list[dict], source: str, teacher_preds: list[dict] | None = None) -> list[dict]:
    """Return ``{"id", "text", "target"}`` rows, with targets from the teacher or the ground truth."""
    if source not in TARGET_SOURCES:
        raise ValueError(f"source must be one of {TARGET_SOURCES}")
    if source == "gold":
        return [{"id": r["id"], "text": r["text"], "target": r["label"]} for r in rows]
    if teacher_preds is None:
        raise ValueError("teacher targets need teacher predictions")
    by_id = {p["id"]: p["pred"].strip() for p in teacher_preds}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} train rows have no teacher prediction, e.g. {missing[:3]}")
    return [{"id": r["id"], "text": r["text"], "target": by_id[r["id"]]} for r in rows]


def nested_subset(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """First ``n`` rows of one seeded shuffle, so smaller subsets are contained in larger ones."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    order = list(rows)
    random.Random(seed).shuffle(order)
    return order[:n]
