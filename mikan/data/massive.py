"""Prepare MASSIVE en-US (intent + slot frames) as Mikan's structured-output task.

    uv run python -m mikan.data.massive --out data/massive

Source: Amazon MASSIVE v1.1 (CC-BY-4.0), FitzGerald et al., 2022. Only the en-US file is
used, and only ``id``, ``utt``, ``annot_utt``, ``intent`` and ``partition`` are kept
(annotator ids are dropped).

Outputs (records are ``{"id", "text", "label", "frame"}``; ``label`` is the intent so the
label-based tooling still works, ``frame`` is the full target):
    intents.txt     60 intents, sorted
    slot_types.txt  55 slot types, sorted
    fewshot.jsonl   ranked teacher-prompt examples (removed from train); every slot type is
                    covered at rank 0
    train.jsonl     distillation inputs (official train minus few-shot, leaks, duplicates)
    val.jsonl       official dev split, shuffled (checkpoint selection; val[:300] is the teacher pilot)
    golden.jsonl    official test split, ground-truth frames, never trained on
    manifest.json   source hash, seed, counts, leakage stats
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from ..frames import grounded, parse_annotation
from .banking77 import _sha256, normalize_text, write_jsonl

SOURCE = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MEMBER = "1.1/data/en-US.jsonl"


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(SOURCE, headers={"User-Agent": "mikan-data"})
        with urllib.request.urlopen(req, timeout=300) as r:
            dest.write_bytes(r.read())
    return dest


def read_records(tar_path: Path) -> list[dict]:
    with tarfile.open(tar_path) as tar:
        lines = tar.extractfile(MEMBER).read().decode("utf-8").splitlines()
    out = []
    for line in lines:
        r = json.loads(line)
        frame = {"intent": r["intent"], "slots": parse_annotation(r["annot_utt"])}
        out.append({"id": f"massive-{int(r['id']):05d}", "text": r["utt"], "label": r["intent"],
                    "frame": frame, "partition": r["partition"]})
    return out


def make_splits(records: list[dict], seed: int = 0, fewshot_per_intent: int = 3) -> tuple[dict[str, list[dict]], dict]:
    by_part: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_part[r.pop("partition")].append(r)
    test = by_part["test"]
    test_texts = {normalize_text(r["text"]) for r in test}

    def clean(rows: list[dict]) -> tuple[list[dict], int, int]:
        seen: set[str] = set()
        kept, leaked, dupes = [], 0, 0
        for r in rows:
            key = normalize_text(r["text"])
            if key in test_texts:
                leaked += 1
            elif key in seen:
                dupes += 1
            else:
                seen.add(key)
                kept.append(r)
        return kept, leaked, dupes

    train, tr_leak, tr_dup = clean(by_part["train"])
    val, va_leak, va_dup = clean(by_part["dev"])
    val_texts = {normalize_text(r["text"]) for r in val}
    before = len(train)
    train = [r for r in train if normalize_text(r["text"]) not in val_texts]
    tr_in_val = before - len(train)

    rng = random.Random(seed)
    pool = train[:]
    rng.shuffle(pool)
    chosen: dict[str, int] = {}  # id -> rank
    # rank 0: one example per uncovered slot type, rarest types first, so every type is shown
    counts: dict[str, int] = defaultdict(int)
    for r in train:
        for s in r["frame"]["slots"]:
            counts[s["type"]] += 1
    covered: set[str] = set()
    for t in sorted(counts, key=lambda t: (counts[t], t)):
        if t in covered:
            continue
        r = next(r for r in pool if r["id"] not in chosen and any(s["type"] == t for s in r["frame"]["slots"]))
        chosen[r["id"]] = 0
        covered |= {s["type"] for s in r["frame"]["slots"]}
    # ranks 0..k-1 per intent, on top of the coverage examples
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_intent[r["label"]].append(r)
    for intent in sorted(by_intent):
        have = sum(1 for r in by_intent[intent] if r["id"] in chosen)
        for r in by_intent[intent]:
            if have >= fewshot_per_intent:
                break
            if r["id"] not in chosen:
                chosen[r["id"]] = have
                have += 1

    fewshot = sorted(({**r, "rank": chosen[r["id"]]} for r in train if r["id"] in chosen), key=lambda r: r["id"])
    train = [r for r in train if r["id"] not in chosen]
    rng.shuffle(val)  # source order is grouped by scenario; val[:n] pilots must be representative
    splits = {"fewshot": fewshot, "train": train, "val": val, "golden": test}
    stats = {
        "train_leaked_into_test_removed": tr_leak, "train_duplicates_removed": tr_dup,
        "train_overlapping_val_removed": tr_in_val,
        "val_leaked_into_test_removed": va_leak, "val_duplicates_removed": va_dup,
    }
    return splits, stats


def prepare(out: Path, seed: int = 0, fewshot_per_intent: int = 3) -> dict:
    tar_path = download(out / "raw" / "amazon-massive-dataset-1.1.tar.gz")
    records = read_records(tar_path)
    intents = sorted({r["label"] for r in records})
    slot_types = sorted({s["type"] for r in records for s in r["frame"]["slots"]})
    splits, stats = make_splits(records, seed, fewshot_per_intent)

    out.mkdir(parents=True, exist_ok=True)
    (out / "intents.txt").write_text("\n".join(intents) + "\n", encoding="utf-8")
    (out / "slot_types.txt").write_text("\n".join(slot_types) + "\n", encoding="utf-8")
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)

    gold = splits["golden"]
    manifest = {
        "dataset": "AmazonScience/massive v1.1 (en-US)",
        "license": "CC-BY-4.0",
        "source": {"url": SOURCE, "member": MEMBER, "sha256": _sha256(tar_path)},
        "seed": seed,
        "fewshot_per_intent": fewshot_per_intent,
        "n_intents": len(intents),
        "n_slot_types": len(slot_types),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "golden_no_slot_share": sum(not r["frame"]["slots"] for r in gold) / len(gold),
        "golden_grounded_rate": sum(grounded(r["frame"], r["text"]) for r in gold) / len(gold),
        **stats,
        "files_sha256": {p.name: _sha256(p) for p in sorted(out.glob("*.jsonl")) + [out / "intents.txt", out / "slot_types.txt"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare MASSIVE en-US splits")
    p.add_argument("--out", default="data/massive")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fewshot-per-intent", type=int, default=3)
    args = p.parse_args(argv)
    m = prepare(Path(args.out), args.seed, args.fewshot_per_intent)
    print(json.dumps({k: v for k, v in m.items() if k not in ("files_sha256", "source")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
