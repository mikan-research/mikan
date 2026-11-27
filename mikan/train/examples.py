"""Training targets and subsets for SFT and GRPO. Standard library only, so it is unit-testable anywhere.

A student is trained on a short prompt (instruction + message, no label list and no
examples). It must learn the label set (or frame schema) itself; the short prompt is what
keeps inference cheap.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

STUDENT_SYSTEM = (
    "Classify the online-banking customer message into exactly one intent. "
    "Reply with the intent label only."
)
STUDENT_FRAMES_SYSTEM = (
    "Parse the voice-assistant request into a JSON frame with its intent and slots. "
    "Reply with the JSON only."
)
SYSTEMS = {"labels": STUDENT_SYSTEM, "frames": STUDENT_FRAMES_SYSTEM}

TARGET_SOURCES = ("teacher", "gold")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def task_kind(task_dir: Path) -> str:
    return "frames" if (Path(task_dir) / "slot_types.txt").exists() else "labels"


def student_messages(text: str, kind: str = "labels") -> list[dict]:
    return [{"role": "system", "content": SYSTEMS[kind]}, {"role": "user", "content": text}]


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


def seeded_order(rows: list[dict], seed: int = 0) -> list[dict]:
    order = list(rows)
    random.Random(seed).shuffle(order)
    return order


def nested_subset(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """First ``n`` rows of one seeded shuffle, so smaller subsets are contained in larger ones."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    return seeded_order(rows, seed)[:n]


def attach_frame_targets(rows: list[dict], source: str, teacher_preds: list[dict] | None = None,
                         intents: set[str] | None = None, slot_types: set[str] | None = None) -> tuple[list[dict], int]:
    """Frames version of ``attach_targets``: the target is the frame's canonical JSON string.

    Teacher outputs that do not parse against the schema are dropped (they cannot be a
    training target); the count is returned so it is reported, not hidden.
    """
    from ..frames import parse, to_json

    if source not in TARGET_SOURCES:
        raise ValueError(f"source must be one of {TARGET_SOURCES}")
    if source == "gold":
        return [{"id": r["id"], "text": r["text"], "target": to_json(r["frame"])} for r in rows], 0
    if teacher_preds is None:
        raise ValueError("teacher targets need teacher predictions")
    by_id = {p["id"]: p["pred"] for p in teacher_preds}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} train rows have no teacher prediction, e.g. {missing[:3]}")
    out, dropped = [], 0
    for r in rows:
        frame = parse(by_id[r["id"]], intents, slot_types)
        if frame is None:
            dropped += 1
            continue
        out.append({"id": r["id"], "text": r["text"], "target": to_json(frame)})
    return out, dropped


def load_task(data: Path, task: str, targets: str, teacher_dir: str = "teacher") -> dict:
    """Train / val / golden rows plus targets for ``data/<task>`` (labels or frames)."""
    tdir = Path(data) / task
    kind = task_kind(tdir)
    rows = read_jsonl(tdir / "train.jsonl")
    teacher = read_jsonl(Path(data) / teacher_dir / "train_preds.jsonl") if targets == "teacher" else None
    dropped = 0
    if kind == "frames":
        intents = set((tdir / "intents.txt").read_text().split())
        types = set((tdir / "slot_types.txt").read_text().split())
        train, dropped = attach_frame_targets(rows, targets, teacher, intents, types)
    else:
        train = attach_targets(rows, targets, teacher)
    return {"kind": kind, "train": train, "dropped_invalid_targets": dropped,
            "val": read_jsonl(tdir / "val.jsonl"), "golden": read_jsonl(tdir / "golden.jsonl")}
