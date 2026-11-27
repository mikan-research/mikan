import json

import numpy as np
import pytest

from mikan import frames as fr
from mikan.eval.data import INVALID, WRONG, align_frames
from mikan.eval.report import evaluate, render_markdown
from mikan.tasks import TaskConfig

INTENTS = ["alarm_set", "weather_query"]
TYPES = ["time", "date", "place_name"]
GOLD = {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "friday"}]}


def j(obj) -> str:
    return json.dumps(obj)


def test_parse_annotation():
    assert fr.parse_annotation("wake me up at [time : nine am] on [date : friday]") == GOLD["slots"]
    assert fr.parse_annotation("quiet") == []
    two = fr.parse_annotation("from [date : monday] to [date : friday]")
    assert [s["value"] for s in two] == ["monday", "friday"]


@pytest.mark.parametrize("raw", [
    None, "", "alarm_set", "```json\n{}\n```", j([GOLD]),
    j({"intent": "alarm_set"}),
    j({**GOLD, "extra": 1}),
    j({"intent": "nope", "slots": []}),
    j({"intent": "alarm_set", "slots": [{"type": "colour", "value": "red"}]}),
    j({"intent": "alarm_set", "slots": [{"type": "time", "value": "  "}]}),
    j({"intent": "alarm_set", "slots": [{"type": "time"}]}),
])
def test_parse_rejects_invalid(raw):
    assert fr.parse(raw, set(INTENTS), set(TYPES)) is None


def test_exact_match_ignores_order_case_and_spacing():
    pred = {"intent": "alarm_set", "slots": [{"type": "date", "value": "Friday"}, {"type": "time", "value": "nine  am"}]}
    assert fr.exact_match(fr.parse(j(pred), set(INTENTS), set(TYPES)), GOLD)
    assert fr.canonical(pred) == fr.canonical(GOLD)
    assert not fr.exact_match({**GOLD, "intent": "weather_query"}, GOLD)
    assert not fr.exact_match({"intent": "alarm_set", "slots": GOLD["slots"][:1]}, GOLD)
    # repeated slot types are a multiset, not a set
    rep = {"intent": "alarm_set", "slots": GOLD["slots"] + GOLD["slots"][:1]}
    assert not fr.exact_match(rep, GOLD)


def test_slot_f1_and_grounding():
    half = {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "monday"}]}
    assert fr.slot_f1(half, GOLD) == 0.5
    assert fr.slot_f1({"intent": "x", "slots": []}, {"intent": "x", "slots": []}) == 1.0
    assert fr.slot_f1(None, {"intent": "x", "slots": []}) == 0.0
    text = "wake me up at nine am on friday"
    assert fr.grounded(GOLD, text) == 1.0 and fr.grounded(half, text) == 0.5


def test_reward():
    text = "wake me up at nine am on friday"
    kw = dict(ref=GOLD, text=text, intents=set(INTENTS), slot_types=set(TYPES))
    assert fr.reward("not json", **kw) == 0.0
    assert fr.reward(j(GOLD), **kw) == pytest.approx(1.0)
    wrong_intent = fr.reward(j({**GOLD, "intent": "weather_query"}), **kw)
    assert wrong_intent == pytest.approx(0.7)
    # a correct-looking but invented span earns nothing for slots
    invented = {"intent": "alarm_set", "slots": [{"type": "time", "value": "ten pm"}]}
    assert fr.reward(j(invented), **kw) == pytest.approx(0.4)
    assert 0 < fr.reward(j({"intent": "alarm_set", "slots": []}), **kw) < wrong_intent


def test_json_schema_shape():
    s = fr.json_schema(INTENTS, TYPES)
    assert s["properties"]["intent"]["enum"] == INTENTS
    assert s["properties"]["slots"]["items"]["properties"]["type"]["enum"] == TYPES


def _gold_rows():
    return [
        {"id": "a", "text": "wake me up at nine am on friday", "label": "alarm_set", "frame": GOLD},
        {"id": "b", "text": "weather in paris", "label": "weather_query",
         "frame": {"intent": "weather_query", "slots": [{"type": "place_name", "value": "paris"}]}},
        {"id": "c", "text": "is it raining", "label": "weather_query", "frame": {"intent": "weather_query", "slots": []}},
    ]


def test_align_frames_codes_items():
    gold = _gold_rows()
    preds = {"s": [
        {"id": "a", "pred": j(GOLD)},
        {"id": "b", "pred": j({"intent": "weather_query", "slots": []})},
        {"id": "c", "pred": "weather_query"},
    ]}
    al = align_frames(gold, preds, INTENTS, TYPES)
    assert al.preds["s"].tolist() == [0, WRONG, INVALID]
    s = al.frame_summary["s"]
    assert s["exact_match"] == pytest.approx(1 / 3) and s["validity_rate"] == pytest.approx(2 / 3)
    assert s["intent_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="ids do not match"):
        align_frames(gold, {"s": preds["s"][:2]}, INTENTS, TYPES)


def test_frames_report(tmp_path):
    (tmp_path / "intents.txt").write_text("\n".join(INTENTS) + "\n")
    (tmp_path / "types.txt").write_text("\n".join(TYPES) + "\n")
    cfg = tmp_path / "t.yaml"
    cfg.write_text("name: t\nkind: frames\nmetric: accuracy\ndelta: 0.03\nn_boot: 200\n"
                   "labels_file: intents.txt\nslot_types_file: types.txt\n")
    task = TaskConfig.load(cfg)
    gold = _gold_rows() * 20
    gold = [{**r, "id": f"{r['id']}{i}"} for i, r in enumerate(gold)]
    perfect = [{"id": r["id"], "pred": fr.to_json(r["frame"])} for r in gold]
    from mikan.eval.data import align_for_task
    al = align_for_task(task, gold, {"teacher": perfect, "student": perfect})
    r = evaluate(al, task)
    assert r["kind"] == "frames" and r["decision"] == "PASS"
    assert r["summary"]["student"]["exact_match"] == 1.0
    md = render_markdown(r)
    assert "Exact match" in md and "Slot F1" in md


def test_frames_config_validation(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("name: t\nkind: frames\nmetric: macro_f1\ndelta: 0.03\nslot_types_file: x\n")
    with pytest.raises(ValueError, match="exact frame match"):
        TaskConfig.load(cfg)
    cfg.write_text("name: t\nkind: frames\nmetric: accuracy\ndelta: 0.03\n")
    with pytest.raises(ValueError, match="slot_types_file"):
        TaskConfig.load(cfg)


def test_massive_task_config_and_data():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    task = TaskConfig.load(repo / "tasks" / "massive.yaml")
    assert task.kind == "frames" and task.delta == 0.03
    golden = repo / "data" / "massive" / "golden.jsonl"
    if not golden.exists():
        pytest.skip("MASSIVE not prepared")
    rows = [json.loads(line) for line in golden.read_text().splitlines()]
    al = align_frames(rows, {"gold": [{"id": r["id"], "pred": fr.to_json(r["frame"])} for r in rows]},
                      task.labels_file.read_text().split(), task.slot_types_file.read_text().split())
    assert al.frame_summary["gold"]["exact_match"] == 1.0 and len(rows) == 2974
    assert np.all(al.preds["gold"] == al.gold)


def test_retrieval_excludes_self_and_held_out(tmp_path):
    from mikan.data.retrieve import nearest
    from mikan.prompts import retrieved_systems

    pool = [{"id": f"t{i}", "text": t, "label": "alarm_set", "frame": {"intent": "alarm_set", "slots": []}}
            for i, t in enumerate(["wake me at nine", "wake me at ten", "play some jazz", "weather in paris"])]
    queries = pool[:1] + [{"id": "v0", "text": "wake me at nine please", "label": "alarm_set",
                           "frame": {"intent": "alarm_set", "slots": []}}]
    demos = nearest(pool, queries, k=2)
    assert "t0" not in demos[0] and demos[0][0] == "t1"
    assert demos[1][0] == "t0"
    for name, rows in (("train", pool), ("fewshot", [])):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "intents.txt").write_text("alarm_set\n")
    (tmp_path / "slot_types.txt").write_text("time\n")
    (tmp_path / "retrieved.jsonl").write_text("".join(json.dumps({"id": q["id"], "demo_ids": d}) + "\n"
                                                      for q, d in zip(queries, demos)))
    systems = retrieved_systems(tmp_path, queries, 2)
    # most similar demo sits last, right before the query
    assert systems[1].rstrip().endswith(json.dumps({"intent": "alarm_set", "slots": []}))
    assert systems[1].index("wake me at nine") > systems[1].index("Examples:")
