"""Prompt construction shared by the teacher and the students. Standard library only."""

from __future__ import annotations

import hashlib

INSTRUCTION = (
    "You classify customer messages sent to an online bank into exactly one intent.\n"
    "Reply with the intent label only, exactly as written in the list, with no other text."
)


def system_prompt(labels: list[str], fewshot: list[dict] | None = None) -> str:
    """Instruction, the full label list, and optional ``{"text", "label"}`` examples.

    The prompt is identical for every query, so vLLM's prefix cache computes it once.
    """
    parts = [INSTRUCTION, "", "Intents:", *(f"- {lab}" for lab in labels)]
    if fewshot:
        parts += ["", "Examples:"]
        for ex in fewshot:
            text = " ".join(ex["text"].split())
            parts += [f"Message: {text}", f"Intent: {ex['label']}", ""]
    return "\n".join(parts).rstrip() + "\n"


FRAME_INSTRUCTION = (
    "You parse requests made to a voice assistant into a semantic frame.\n"
    "Return one JSON object: {\"intent\": <one intent from the list>, \"slots\": [{\"type\": <slot type from the list>, "
    "\"value\": <the exact words from the request>}, ...]}.\n"
    "Slot values must be copied verbatim from the request. List slots in the order they appear; "
    "a type may repeat. Use an empty list when the request has no slots. Output the JSON only."
)


def frames_system_prompt(intents: list[str], slot_types: list[str], fewshot: list[dict] | None = None) -> str:
    """Instruction, intent and slot-type lists, and optional ``{"text", "frame"}`` examples."""
    import json

    parts = [FRAME_INSTRUCTION, "", "Intents: " + ", ".join(intents), "", "Slot types: " + ", ".join(slot_types)]
    if fewshot:
        parts += ["", "Examples:"]
        for ex in fewshot:
            text = " ".join(ex["text"].split())
            parts += [f"Request: {text}", "Frame: " + json.dumps(ex["frame"], ensure_ascii=False), ""]
    return "\n".join(parts).rstrip() + "\n"


def select_shots(pool: list[dict], shots: int) -> list[dict] | None:
    """Examples with ``rank < shots`` (all ranks of a label are kept together), or None for zero-shot."""
    if shots <= 0:
        return None
    chosen = [r for r in pool if r.get("rank", 0) < shots]
    return sorted(chosen, key=lambda r: (r.get("rank", 0), r["label"]))


def prompt_sha256(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()


def task_setup(data_dir, shots: int, constrained: bool = True) -> dict:
    """Prompt and decoding constraint for a prepared data dir (labels or frames task)."""
    import json
    from pathlib import Path

    data = Path(data_dir)

    def lines(name: str) -> list[str]:
        return [ln.strip() for ln in (data / name).read_text(encoding="utf-8").splitlines() if ln.strip()]

    pool = [json.loads(ln) for ln in (data / "fewshot.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    shots_pool = select_shots(pool, shots)
    if (data / "slot_types.txt").exists():
        from .frames import json_schema

        intents, types = lines("intents.txt"), lines("slot_types.txt")
        return {"kind": "frames", "system": frames_system_prompt(intents, types, shots_pool), "choices": None,
                "json_schema": json_schema(intents, types) if constrained else None, "max_tokens": 160}
    labels = lines("labels.txt")
    return {"kind": "labels", "system": system_prompt(labels, shots_pool),
            "choices": labels if constrained else None, "json_schema": None, "max_tokens": 24}


def retrieved_systems(data_dir, rows: list[dict], k: int) -> list[str]:
    """Per-item frames prompts: the shared instruction and lists, then the item's ``k`` nearest
    labelled examples (most similar last, next to the query). Needs ``retrieved.jsonl``."""
    import json
    from pathlib import Path

    data = Path(data_dir)

    def read(name: str) -> list[dict]:
        return [json.loads(x) for x in (data / name).read_text(encoding="utf-8").splitlines() if x.strip()]

    intents = (data / "intents.txt").read_text(encoding="utf-8").split()
    types = (data / "slot_types.txt").read_text(encoding="utf-8").split()
    pool = {r["id"]: r for r in read("train.jsonl") + read("fewshot.jsonl")}
    demos = {r["id"]: r["demo_ids"] for r in read("retrieved.jsonl")}
    out = []
    for r in rows:
        ids = demos[r["id"]][:k]
        if r["id"] in ids:
            raise ValueError(f"{r['id']} retrieves itself")
        out.append(frames_system_prompt(intents, types, [pool[i] for i in reversed(ids)]))
    return out
