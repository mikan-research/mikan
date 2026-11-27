"""Loading gold labels and predictions, and aligning them by item id.

File formats (JSON Lines):
    gold:        {"id": ..., "label": "<label string>", "text": "<optional input>"}
                 frames tasks add "frame": {"intent", "slots"}; "label" is the intent
    predictions: {"id": ..., "pred": "<raw model output>"}

For frames tasks each prediction is coded per item as the gold intent index when the
whole frame matches exactly, INVALID when it does not parse against the schema, and
WRONG otherwise, so every label-based statistic becomes an exact-match statistic.

Every gold id must have exactly one prediction in every file, and no file may
contain unknown ids. A mismatch raises instead of silently scoring a subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import frames as fr
from .metrics import INVALID

WRONG = -2  # a valid frame that does not match gold


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from None
    return records


def load_labels(path: str | Path) -> list[str]:
    labels = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: duplicate labels")
    if len(labels) < 2:
        raise ValueError(f"{path}: need at least 2 labels")
    return labels


def normalize_pred(raw) -> str:
    """Strict normalisation: only surrounding whitespace is forgiven."""
    return "" if raw is None else str(raw).strip()


@dataclass
class Aligned:
    labels: list[str]
    ids: list[str]
    gold: np.ndarray
    preds: dict[str, np.ndarray]
    raw_preds: dict[str, list[str]]
    texts: list[str | None] = field(default_factory=list)
    gold_display: list[str] | None = None  # how to show gold in examples (frames: canonical frame)
    frame_summary: dict[str, dict] | None = None  # frames only: per-system frame metrics

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def _index_by_id(records: list[dict], key: str, source: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        if "id" not in r or key not in r:
            raise ValueError(f"{source}: every record needs 'id' and '{key}'")
        rid = str(r["id"])
        if rid in out:
            raise ValueError(f"{source}: duplicate id {rid!r}")
        out[rid] = r
    return out


def align(gold_records: list[dict], pred_records: dict[str, list[dict]], labels: list[str]) -> Aligned:
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    gold_by_id = _index_by_id(gold_records, "label", "gold")
    ids = list(gold_by_id)

    gold = np.empty(len(ids), dtype=np.int64)
    texts: list[str | None] = []
    for i, rid in enumerate(ids):
        lab = str(gold_by_id[rid]["label"]).strip()
        if lab not in label_to_id:
            raise ValueError(f"gold: id {rid!r} has unknown label {lab!r}")
        gold[i] = label_to_id[lab]
        texts.append(gold_by_id[rid].get("text"))

    preds: dict[str, np.ndarray] = {}
    raw_preds: dict[str, list[str]] = {}
    gold_ids = set(ids)
    for name, records in pred_records.items():
        by_id = _index_by_id(records, "pred", name)
        missing = gold_ids - by_id.keys()
        extra = by_id.keys() - gold_ids
        if missing or extra:
            raise ValueError(
                f"{name}: ids do not match gold "
                f"(missing {len(missing)}, e.g. {sorted(missing)[:5]}; extra {len(extra)}, e.g. {sorted(extra)[:5]})"
            )
        raw = [normalize_pred(by_id[rid]["pred"]) for rid in ids]
        raw_preds[name] = raw
        preds[name] = np.array([label_to_id.get(p, INVALID) for p in raw], dtype=np.int64)

    return Aligned(labels=labels, ids=ids, gold=gold, preds=preds, raw_preds=raw_preds, texts=texts)


def _check_ids(name: str, by_id: dict, gold_ids: set) -> None:
    missing = gold_ids - by_id.keys()
    extra = by_id.keys() - gold_ids
    if missing or extra:
        raise ValueError(
            f"{name}: ids do not match gold "
            f"(missing {len(missing)}, e.g. {sorted(missing)[:5]}; extra {len(extra)}, e.g. {sorted(extra)[:5]})"
        )


def align_frames(gold_records: list[dict], pred_records: dict[str, list[dict]],
                 intents: list[str], slot_types: list[str]) -> Aligned:
    intent_to_id = {lab: i for i, lab in enumerate(intents)}
    intent_set, type_set = set(intents), set(slot_types)
    gold_by_id = _index_by_id(gold_records, "frame", "gold")
    ids = list(gold_by_id)
    gold_frames = []
    for rid in ids:
        f = fr.parse(fr.to_json(gold_by_id[rid]["frame"]), intent_set, type_set)
        if f is None:
            raise ValueError(f"gold: id {rid!r} has a frame outside the schema")
        gold_frames.append(f)
    gold = np.array([intent_to_id[f["intent"]] for f in gold_frames], dtype=np.int64)
    texts = [gold_by_id[rid].get("text") for rid in ids]

    preds, raw_preds, summaries = {}, {}, {}
    for name, records in pred_records.items():
        by_id = _index_by_id(records, "pred", name)
        _check_ids(name, by_id, set(ids))
        raw = [normalize_pred(by_id[rid]["pred"]) for rid in ids]
        parsed = [fr.parse(x, intent_set, type_set) for x in raw]
        code = [INVALID if p is None else (int(gold[i]) if fr.exact_match(p, gold_frames[i]) else WRONG)
                for i, p in enumerate(parsed)]
        preds[name] = np.array(code, dtype=np.int64)
        raw_preds[name] = raw
        summaries[name] = fr.summary(gold_frames, parsed, [t or "" for t in texts])
    return Aligned(labels=intents, ids=ids, gold=gold, preds=preds, raw_preds=raw_preds, texts=texts,
                   gold_display=[fr.to_json(f) for f in gold_frames], frame_summary=summaries)


def align_for_task(task, gold_records: list[dict], pred_records: dict[str, list[dict]],
                   labels_path: str | Path | None = None) -> Aligned:
    """Align with the method the task config asks for (``kind: labels`` or ``kind: frames``)."""
    labels_path = labels_path or task.labels_file
    if labels_path is None:
        raise ValueError("no labels file (set labels_file in the task config or pass --labels)")
    labels = load_labels(labels_path)
    if task.kind == "frames":
        return align_frames(gold_records, pred_records, labels, load_labels(task.slot_types_file))
    return align(gold_records, pred_records, labels)
