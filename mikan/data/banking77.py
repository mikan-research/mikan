"""Prepare Banking77: labels, few-shot pool, train/val splits and the golden test set.

    uv run python -m mikan.data.banking77 --out data/banking77

Outputs (JSON Lines records are ``{"id", "text", "label"}``):
    labels.txt      77 label names, one per line, in dataset order
    fewshot.jsonl   ranked prompt examples per label (removed from train); "rank" 0..k-1
    train.jsonl     distillation training inputs
    val.jsonl       checkpoint selection, teacher pilot
    golden.jsonl    official test split, ground-truth labels, never trained on
    manifest.json   source hashes, seed, counts, leakage stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

SOURCE = "https://huggingface.co/api/datasets/PolyAI/banking77/parquet/default/{split}/0.parquet"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(split: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(SOURCE.format(split=split), headers={"User-Agent": "mikan-data"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def read_parquet(path: Path) -> tuple[list[str], list[int], list[str]]:
    """Return ``(texts, label_ids, label_names)``; label names come from the HF schema metadata."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    return table.column("text").to_pylist(), table.column("label").to_pylist(), names


def make_splits(
    train: list[dict],
    test: list[dict],
    seed: int = 0,
    val_frac: float = 0.1,
    fewshot_per_label: int = 1,
) -> tuple[dict[str, list[dict]], dict]:
    """Split the official train set into val / fewshot / train, stratified by label.

    Per label, the shuffled items are cut in this order: validation (a fixed
    ``val_frac`` share of the label's items), then ``fewshot_per_label`` ranked
    prompt examples, then train. The validation set therefore does not depend on
    the few-shot pool size, and the k-shot pool is a prefix of the (k+1)-shot pool.

    Train items whose normalised text also appears in the test set are dropped
    (leakage), as are exact duplicates within train. The test set is kept whole
    as the golden set.
    """
    test_texts = {normalize_text(r["text"]) for r in test}
    seen: set[str] = set()
    clean: list[dict] = []
    leaked = dupes = 0
    for r in train:
        key = normalize_text(r["text"])
        if key in test_texts:
            leaked += 1
        elif key in seen:
            dupes += 1
        else:
            seen.add(key)
            clean.append(r)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in clean:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"fewshot": [], "train": [], "val": []}
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n_val = max(1, round(val_frac * len(items)))
        if len(items) < n_val + fewshot_per_label + 1:
            raise ValueError(f"label {label!r} has too few items ({len(items)}) to split")
        splits["val"] += items[:n_val]
        shots = items[n_val : n_val + fewshot_per_label]
        splits["fewshot"] += [{**r, "rank": k} for k, r in enumerate(shots)]
        splits["train"] += items[n_val + fewshot_per_label :]

    for name in splits:  # deterministic, label-interleaved order
        splits[name].sort(key=lambda r: r["id"])
    splits["golden"] = list(test)
    stats = {"train_leaked_into_test_removed": leaked, "train_duplicates_removed": dupes}
    return splits, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(out: Path, seed: int = 0, val_frac: float = 0.1, fewshot_per_label: int = 5) -> dict:
    raw = out / "raw"
    sources = {}
    records: dict[str, list[dict]] = {}
    labels: list[str] | None = None
    for split in ("train", "test"):
        path = download(split, raw / f"{split}.parquet")
        sources[split] = {"url": SOURCE.format(split=split), "sha256": _sha256(path)}
        texts, ids, names = read_parquet(path)
        if labels is None:
            labels = names
        elif names != labels:
            raise ValueError("train and test label names differ")
        records[split] = [
            {"id": f"{split}-{i:05d}", "text": t, "label": labels[k]} for i, (t, k) in enumerate(zip(texts, ids))
        ]

    splits, stats = make_splits(records["train"], records["test"], seed, val_frac, fewshot_per_label)
    (out / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)

    manifest = {
        "dataset": "PolyAI/banking77",
        "license": "CC-BY-4.0",
        "sources": sources,
        "seed": seed,
        "val_frac": val_frac,
        "fewshot_per_label": fewshot_per_label,
        "n_labels": len(labels),
        "counts": {name: len(rows) for name, rows in splits.items()},
        **stats,
        "files_sha256": {
            p.name: _sha256(p) for p in sorted(out.glob("*.jsonl")) + [out / "labels.txt"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare Banking77 splits")
    p.add_argument("--out", default="data/banking77")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--fewshot-per-label", type=int, default=5)
    args = p.parse_args(argv)
    m = prepare(Path(args.out), args.seed, args.val_frac, args.fewshot_per_label)
    print(json.dumps({k: m[k] for k in ("counts", "train_leaked_into_test_removed", "train_duplicates_removed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
