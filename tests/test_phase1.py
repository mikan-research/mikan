import json

import pytest

from mikan.data.banking77 import make_splits, normalize_text
from mikan.intake.runs import load_split, write_run
from mikan.prompts import prompt_sha256, select_shots, system_prompt


def _records(prefix, n_per_label, labels, text=lambda lab, i: f"{lab} message {i}"):
    return [
        {"id": f"{prefix}-{lab}-{i:03d}", "text": text(lab, i), "label": lab}
        for lab in labels
        for i in range(n_per_label)
    ]


def test_normalize_text():
    assert normalize_text("  Where IS   my\ncard? ") == "where is my card?"


def test_make_splits_removes_leakage_and_duplicates():
    labels = ["a", "b"]
    train = _records("train", 30, labels)
    train.append({"id": "train-dup", "text": "A MESSAGE 0", "label": "a"})  # duplicate after normalising
    test = [{"id": "test-0", "text": "b message 5", "label": "b"}]  # also in train -> leak
    splits, stats = make_splits(train, test, seed=0, val_frac=0.1)
    assert stats == {"train_leaked_into_test_removed": 1, "train_duplicates_removed": 1}
    used = {normalize_text(r["text"]) for s in ("fewshot", "train", "val") for r in splits[s]}
    assert "b message 5" not in used
    assert splits["golden"] == test


def test_make_splits_is_stratified_disjoint_and_deterministic():
    labels = [f"l{k}" for k in range(5)]
    train = _records("train", 41, labels)
    s1, _ = make_splits(train, [], seed=3)
    s2, _ = make_splits(train, [], seed=3)
    s3, _ = make_splits(train, [], seed=4)
    assert s1 == s2 and s1 != s3
    ids = [r["id"] for s in ("fewshot", "train", "val") for r in s1[s]]
    assert len(ids) == len(set(ids)) == len(train)
    for lab in labels:
        assert sum(r["label"] == lab for r in s1["fewshot"]) == 1
        assert sum(r["label"] == lab for r in s1["val"]) == 4  # round(0.1 * 41)


def test_val_split_does_not_depend_on_fewshot_pool_size():
    train = _records("train", 30, ["a", "b", "c"])
    s1, _ = make_splits(train, [], seed=0, fewshot_per_label=1)
    s5, _ = make_splits(train, [], seed=0, fewshot_per_label=5)
    assert s1["val"] == s5["val"]
    one = {r["id"] for r in s1["fewshot"]}
    assert one == {r["id"] for r in s5["fewshot"] if r["rank"] == 0}
    assert len(s5["fewshot"]) == 15 and len(s5["train"]) == len(s1["train"]) - 12


def test_select_shots():
    pool = [{"text": f"t{k}{lab}", "label": lab, "rank": k} for lab in ("a", "b") for k in range(3)]
    assert select_shots(pool, 0) is None
    assert [(r["label"], r["rank"]) for r in select_shots(pool, 2)] == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]


def test_make_splits_rejects_tiny_labels():
    with pytest.raises(ValueError, match="too few"):
        make_splits(_records("train", 2, ["a"]), [])


def test_system_prompt_contents_and_stability():
    labels = ["card_arrival", "age_limit"]
    shots = [{"text": "where is\nmy card", "label": "card_arrival"}]
    p = system_prompt(labels, shots)
    assert "- card_arrival\n- age_limit" in p
    assert "Message: where is my card\nIntent: card_arrival" in p
    assert "Examples" not in system_prompt(labels)
    assert prompt_sha256(p) == prompt_sha256(system_prompt(labels, shots))


def test_load_split_subsets_and_write_run(tmp_path):
    rows = [{"id": f"v{i}", "text": f"t{i}", "label": "a" if i % 2 else "b"} for i in range(400)]
    (tmp_path / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(load_split(tmp_path, "smoke")) == 50
    assert len(load_split(tmp_path, "pilot")) == 300
    assert len(load_split(tmp_path, "val")) == 400

    sub = rows[:4]
    acc = write_run(tmp_path / "out", "smoke", sub, ["b", "a", "a ", "x"], {"model": "m"})
    assert acc == 0.5  # trailing whitespace is forgiven, 'x' and the wrong 'a' are not
    meta = json.loads((tmp_path / "out" / "smoke_meta.json").read_text())
    assert meta["n"] == 4 and meta["accuracy_vs_gold"] == 0.5 and meta["model"] == "m"
    with pytest.raises(ValueError):
        write_run(tmp_path / "out", "smoke", sub, ["a"], {})
