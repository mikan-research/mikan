"""Loading splits and writing prediction runs. Standard library only (runs inside the Modal client)."""

from __future__ import annotations

import json
from pathlib import Path

# Named subsets: (base split, start, stop). smoke/pilot are cheap checks before a full run;
# train_part1/2 split the MASSIVE teacher run across two Kaggle sessions.
SUBSETS = {
    "smoke": ("val", 0, 50), "pilot": ("val", 0, 300),
    "train_part1": ("train", 0, 8127), "train_part2": ("train", 8127, None),
}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(data_dir: Path, split: str) -> list[dict]:
    base, start, stop = SUBSETS.get(split, (split, 0, None))
    rows = read_jsonl(Path(data_dir) / f"{base}.jsonl")
    return rows[start:stop]


def write_run(out_dir: Path, split: str, rows: list[dict], preds: list[str], meta: dict) -> float:
    """Write ``{split}_preds.jsonl`` and ``{split}_meta.json``; return accuracy (or exact frame match) vs gold."""
    if len(rows) != len(preds):
        raise ValueError(f"{split}: {len(rows)} rows but {len(preds)} predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}_preds.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "pred": p}, ensure_ascii=False) + "\n")
    if rows and "frame" in rows[0]:  # structured-output task: exact frame match
        from ..frames import exact_match, parse

        correct = sum(exact_match(parse(p), r["frame"]) for r, p in zip(rows, preds))
    else:
        correct = sum(p.strip() == r["label"] for r, p in zip(rows, preds))
    acc = correct / len(rows) if rows else float("nan")
    meta = {**meta, "split": split, "n": len(rows), "accuracy_vs_gold": acc}
    (out_dir / f"{split}_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return acc
