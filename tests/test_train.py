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
    g = {"name": "g", "script": "grpo", "adapter": "@runs/sft-a/best_adapter"}
    assert to_flags(g) == ["--adapter", "/kaggle/working/runs/sft-a/best_adapter"]


@pytest.mark.parametrize("configs", ["configs/sft_smoke.json", "configs/sft_phase2.json", "configs/sft_phase2_4b.json",
                                     "configs/massive_grpo_smoke.json", "configs/massive_phase3_a.json", "configs/massive_phase3_b.json"])
def test_configs_parse_into_trainer_arguments(configs):
    import importlib

    for cfg in json.loads((REPO / configs).read_text()):
        mod = importlib.import_module(f"mikan.train.{cfg.get('script', 'sft')}")
        args = mod.parse_args(["--data", "d", "--out", "o", *to_flags(cfg)])
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


def test_teacher_bundle(tmp_path):
    if not (REPO / "data" / "massive" / "golden.jsonl").exists():
        pytest.skip("MASSIVE not prepared")
    path = kaggle_job.build_bundle(REPO / "configs" / "teacher_massive_pilot.json", tmp_path, ("massive",), with_teacher=False)
    with tarfile.open(path) as tar:
        names = set(tar.getnames())
    assert {"mikan/intake/kaggle_teacher.py", "mikan/frames.py", "data/massive/slot_types.txt",
            "data/massive/fewshot.jsonl", "data/massive/golden.jsonl"} <= names
    assert not any(n.startswith("data/teacher/") or "raw" in n for n in names)


def test_vllm_cli_args_parse():
    import ast
    src = (REPO / "mikan" / "intake" / "kaggle_teacher.py").read_text()
    ast.parse(src)  # uploaded verbatim as the kernel script
    assert "mikan.intake.vllm_predict" in src


def test_frame_targets_and_task_loading(tmp_path):
    from mikan.frames import parse
    from mikan.train.examples import attach_frame_targets, load_task, seeded_order, student_messages

    frame = {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}]}
    rows = [{"id": "a", "text": "wake me at nine am", "label": "alarm_set", "frame": frame},
            {"id": "b", "text": "quiet", "label": "audio_volume_mute", "frame": {"intent": "audio_volume_mute", "slots": []}}]
    gold, dropped = attach_frame_targets(rows, "gold")
    assert dropped == 0 and parse(gold[0]["target"]) == frame
    teacher = [{"id": "a", "pred": json.dumps(frame)}, {"id": "b", "pred": "not json"}]
    t, dropped = attach_frame_targets(rows, "teacher", teacher, {"alarm_set", "audio_volume_mute"}, {"time"})
    assert dropped == 1 and [r["id"] for r in t] == ["a"]

    d = tmp_path / "massive"
    d.mkdir()
    for name in ("train", "val", "golden"):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "intents.txt").write_text("alarm_set\naudio_volume_mute\n")
    (d / "slot_types.txt").write_text("time\n")
    task = load_task(tmp_path, "massive", "gold")
    assert task["kind"] == "frames" and len(task["train"]) == 2
    assert "JSON" in student_messages("x", "frames")[0]["content"]
    assert seeded_order(ROWS, 3)[:10] == nested_subset(ROWS, 10, seed=3)


def test_grpo_rewards():
    from mikan.train.grpo import make_rewards

    frame_r, valid_r, grounded_r = make_rewards({"alarm_set"}, {"time"})
    ref = json.dumps({"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}]})
    good = ref
    invented = json.dumps({"intent": "alarm_set", "slots": [{"type": "time", "value": "ten pm"}]})
    kw = dict(ref=[ref] * 3, text=["wake me at nine am"] * 3)
    r = frame_r([good, invented, "garbage"], **kw)
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(0.4) and r[2] == 0.0
    assert valid_r([good, invented, "garbage"]) == [1.0, 1.0, 0.0]
    assert grounded_r([good, invented, "garbage"], text=kw["text"]) == [1.0, 0.0, 0.0]
    # conversational completions are accepted too
    assert frame_r([[{"role": "assistant", "content": good}]], ref=[ref], text=["wake me at nine am"]) == [pytest.approx(1.0)]


@pytest.mark.parametrize("module", ["mikan.train.sft", "mikan.train.grpo"])
def test_trainer_cli_parses(module):
    import importlib

    mod = importlib.import_module(module)
    extra = ["--adapter", "a"] if module.endswith("grpo") else []
    args = mod.parse_args(["--data", "d", "--out", "o", "--model", "m", *extra])
    assert args.out == "o"


def test_postprocess_matches_reward_for_frames():
    from mikan.frames import parse
    from mikan.train.gen import postprocess  # unsloth is imported only inside greedy()

    multi = '{"intent": "alarm_set",\n "slots": []}\n'
    assert parse(postprocess(multi, first_line=False)) is not None   # frames: whole output, as the reward sees it
    assert parse(postprocess(multi, first_line=True)) is None        # the old first-line rule would cut it
    assert postprocess("  card_arrival \nextra", first_line=True) == "card_arrival"  # labels unchanged
