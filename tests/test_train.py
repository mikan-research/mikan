import json
import tarfile
from pathlib import Path

import pytest

from mikan.train import kaggle_job
from mikan.train.examples import STUDENT_SYSTEM, attach_targets, nested_subset, student_messages
from mikan.train.kaggle_runner import to_flags

REPO = Path(__file__).resolve().parents[1]
ROWS = [{"id": f"train-{i:05d}", "text": f"msg {i}", "label": f"l{i % 3}"} for i in range(50)]


def test_attach_gold_and_teacher_targets():
    gold = attach_targets(ROWS, "gold")
    assert gold[0] == {"id": "train-00000", "text": "msg 0", "target": "l0"}
    preds = [{"id": r["id"], "pred": " teach " } for r in ROWS]
    assert {r["target"] for r in attach_targets(ROWS, "teacher", preds)} == {"teach"}


def test_attach_teacher_requires_complete_predictions():
    with pytest.raises(ValueError, match="no teacher prediction"):
        attach_targets(ROWS, "teacher", [{"id": "train-00000", "pred": "x"}])
    with pytest.raises(ValueError):
        attach_targets(ROWS, "teacher")
    with pytest.raises(ValueError):
        attach_targets(ROWS, "noise")


def test_nested_subsets():
    small, big = nested_subset(ROWS, 10, seed=1), nested_subset(ROWS, 25, seed=1)
    assert len(small) == 10 and len(big) == 25
    assert {r["id"] for r in small} <= {r["id"] for r in big}
    assert nested_subset(ROWS, 0) == ROWS and nested_subset(ROWS, 999) == ROWS


def test_student_prompt_is_short():
    msgs = student_messages("where is my card")
    assert msgs[0]["content"] == STUDENT_SYSTEM and msgs[1]["content"] == "where is my card"
    assert len(STUDENT_SYSTEM) < 200  # no label list, no examples


def test_runner_flags():
    cfg = {"name": "x", "gpu": 1, "model": "Qwen/Qwen3-1.7B", "n_train": 500, "load_in_4bit": True, "eval_limit": None}
    assert to_flags(cfg) == ["--model", "Qwen/Qwen3-1.7B", "--n-train", "500", "--load-in-4bit"]


@pytest.mark.parametrize("configs", ["configs/sft_smoke.json", "configs/sft_phase2.json", "configs/sft_phase2_4b.json"])
def test_configs_parse_into_sft_arguments(configs):
    from mikan.train.sft import parse_args

    for cfg in json.loads((REPO / configs).read_text()):
        args = parse_args(["--data", "d", "--out", "o", *to_flags(cfg)])
        assert args.model == cfg["model"] and args.targets == cfg["targets"]


def test_bundle_contents(tmp_path, monkeypatch):
    teacher = REPO / kaggle_job.TEACHER_RUN
    if not (REPO / "data" / "banking77" / "train.jsonl").exists() or not teacher.exists():
        pytest.skip("prepared data and teacher run not present")
    path = kaggle_job.build_bundle(REPO / "configs" / "sft_smoke.json", tmp_path)
    with tarfile.open(path) as tar:
        names = set(tar.getnames())
        info = json.loads(tar.extractfile("bundle.json").read())
    assert {"mikan/train/sft.py", "mikan/intake/runs.py", "configs.json", "data/teacher/train_preds.jsonl",
            "data/banking77/golden.jsonl"} <= names
    assert len(info["commit"]) == 40
