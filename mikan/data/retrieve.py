"""Nearest labelled examples per item, for retrieval-based few-shot teacher prompts.

    uv run python -m mikan.data.retrieve --data data/massive

The pool is every labelled training item (``train`` + ``fewshot``). For each item of
``train``, ``val`` and ``golden`` it writes the ids of the ``--k`` most similar pool items
by TF-IDF cosine over word 1-2 grams to ``retrieved.jsonl`` (``{"id", "demo_ids"}``,
most similar first). An item never retrieves itself, and the pool never contains val or
golden items, so no evaluation label reaches a prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[a-z0-9']+")


def ngrams(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def tfidf(docs: list[list[str]], vocab: dict[str, int] | None = None, idf: np.ndarray | None = None):
    """Row-normalised TF-IDF matrix; fit vocab and idf when not given."""
    if vocab is None:
        df = Counter(t for d in docs for t in set(d))
        vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
    rows, cols, vals = [], [], []
    for i, d in enumerate(docs):
        for t, c in Counter(d).items():
            j = vocab.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append((1 + math.log(c)) * idf[j])
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)))
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return sparse.diags(1 / norms) @ m, vocab, idf


def nearest(pool: list[dict], queries: list[dict], k: int, chunk: int = 2048) -> list[list[str]]:
    pm, vocab, idf = tfidf([ngrams(r["text"]) for r in pool])
    qm, _, _ = tfidf([ngrams(r["text"]) for r in queries], vocab, idf)
    pool_ids = np.array([r["id"] for r in pool])
    out: list[list[str]] = []
    for s in range(0, len(queries), chunk):
        sims = (qm[s : s + chunk] @ pm.T).toarray()
        for i, row in enumerate(sims):
            qid = queries[s + i]["id"]
            top = np.argsort(-row, kind="stable")[: k + 1]
            ids = [pid for pid in pool_ids[top] if pid != qid][:k]  # never the item itself
            out.append(ids)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrieve nearest labelled examples per item")
    p.add_argument("--data", default="data/massive")
    p.add_argument("--k", type=int, default=32)
    args = p.parse_args(argv)
    data = Path(args.data)

    def read(name: str) -> list[dict]:
        return [json.loads(x) for x in (data / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    pool = read("train") + read("fewshot")
    held_out = {r["id"] for r in read("val")} | {r["id"] for r in read("golden")}
    assert not held_out & {r["id"] for r in pool}, "val/golden items in the retrieval pool"
    queries = read("train") + read("val") + read("golden")
    demos = nearest(pool, queries, args.k)
    with open(data / "retrieved.jsonl", "w", encoding="utf-8") as f:
        for q, ids in zip(queries, demos):
            f.write(json.dumps({"id": q["id"], "demo_ids": ids}) + "\n")
    print(f"wrote {len(queries)} rows with {args.k} demos each to {data / 'retrieved.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
