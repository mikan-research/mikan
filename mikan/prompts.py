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


def select_shots(pool: list[dict], shots: int) -> list[dict] | None:
    """Examples with ``rank < shots`` (all ranks of a label are kept together), or None for zero-shot."""
    if shots <= 0:
        return None
    chosen = [r for r in pool if r.get("rank", 0) < shots]
    return sorted(chosen, key=lambda r: (r.get("rank", 0), r["label"]))


def prompt_sha256(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()
