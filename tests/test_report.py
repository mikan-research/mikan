import json
from pathlib import Path

import pytest

from mikan.eval.cli import main
from mikan.eval.data import align, load_labels
from mikan.eval.synthetic import make_paired
from mikan.tasks import TaskConfig

REPO = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def run_files(tmp_path):
    labels = [f"intent_{k}" for k in range(12)]
    (tmp_path / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    gold, t, s = make_paired(1500, 12, 0.9, 0.08, 0.0, seed=11)
    _write_jsonl(tmp_path / "gold.jsonl", [{"id": i, "label": labels[g], "text": f"query {i}"} for i, g in enumerate(gold)])
    _write_jsonl(tmp_path / "teacher.jsonl", [{"id": i, "pred": labels[p]} for i, p in enumerate(t)])
    student = [{"id": i, "pred": f"  {labels[p]} "} for i, p in enumerate(s)]  # whitespace is forgiven
    student[0]["pred"] = "Sure! The intent is intent_3"  # invalid output
    _write_jsonl(tmp_path / "student.jsonl", student)
    task = tmp_path / "task.yaml"
    task.write_text(
        "name: synthetic\nmetric: accuracy\nsecondary_metrics: [macro_f1]\ndelta: 0.03\n"
        "n_boot: 2000\nlabels_file: labels.txt\nlabel_source: synthetic ground truth\n",
        encoding="utf-8",
    )
    return tmp_path


def test_cli_report_end_to_end(run_files, capsys):
    d = run_files
    code = main([
        "report", "--task", str(d / "task.yaml"), "--gold", str(d / "gold.jsonl"),
        "--a", str(d / "teacher.jsonl"), "--b", str(d / "student.jsonl"), "--out", str(d / "out"),
    ])
    assert code == 0
    assert capsys.readouterr().out.startswith("PASS")
    r = json.loads((d / "out" / "report.json").read_text())
    assert r["decision"] == "PASS"
    assert r["n_items"] == 1500 and r["n_classes"] == 12
    assert r["summary"]["student"]["validity_rate"] == pytest.approx(1 - 1 / 1500)
    assert r["primary"]["high"] < 0.03
    assert len(r["config"]["sha256"]) == 64
    assert any("invalid outputs" in x for x in r["limitations"])
    md = (d / "out" / "report.md").read_text()
    assert "## Decision: PASS" in md and "## Per-label slices" in md


def test_align_rejects_missing_and_extra_ids():
    labels = ["a", "b"]
    gold = [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]
    with pytest.raises(ValueError, match="missing 1"):
        align(gold, {"student": [{"id": 1, "pred": "a"}]}, labels)
    with pytest.raises(ValueError, match="extra 1"):
        align(gold, {"student": [{"id": 1, "pred": "a"}, {"id": 2, "pred": "b"}, {"id": 3, "pred": "a"}]}, labels)


def test_align_rejects_unknown_gold_label_and_duplicates():
    with pytest.raises(ValueError, match="unknown label"):
        align([{"id": 1, "label": "zzz"}], {}, ["a", "b"])
    with pytest.raises(ValueError, match="duplicate id"):
        align([{"id": 1, "label": "a"}, {"id": 1, "label": "b"}], {}, ["a", "b"])


def test_load_labels_rejects_duplicates(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("a\nb\na\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_labels(p)


def test_banking77_config_is_valid():
    cfg = TaskConfig.load(REPO / "tasks" / "banking77.yaml")
    assert cfg.metric == "accuracy" and cfg.delta == 0.02
    assert cfg.labels_file == (REPO / "data" / "banking77" / "labels.txt").resolve()


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("name: x\nmetric: accuracy\ndelta: 0.02\nmargin: 0.01\n")
    with pytest.raises(ValueError, match="unknown keys"):
        TaskConfig.load(p)


def test_cli_power(capsys):
    assert main(["power", "--n", "3080", "--disagreement", "0.08", "--delta", "0.02", "--sims", "2000"]) == 0
    out = capsys.readouterr().out
    assert "analytic power" in out and "required n" in out
