"""Loading gold labels and predictions, and aligning them by item id.

File formats (JSON Lines):
    gold:        {"id": ..., "label": "<label string>", "text": "<optional input>"}
    predictions: {"id": ..., "pred": "<raw model output>"}

Every gold id must have exactly one prediction in every file, and no file may
contain unknown ids. A mismatch raises instead of silently scoring a subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import INVALID


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
