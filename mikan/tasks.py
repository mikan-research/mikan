"""Task configs: the pre-registered evaluation settings for one task."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from .eval.metrics import METRICS


@dataclass(frozen=True)
class TaskConfig:
    name: str
    metric: str
    delta: float
    confidence: float = 0.95
    power: float = 0.8
    n_boot: int = 10_000
    seed: int = 0
    secondary_metrics: tuple[str, ...] = ()
    labels_file: Path | None = None
    label_source: str = "unspecified"
    kind: str = "labels"
    slot_types_file: Path | None = None
    path: Path | None = None
    sha256: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        path = Path(path)
        raw_bytes = path.read_bytes()
        cfg = yaml.safe_load(raw_bytes) or {}
        known = {
            "name", "metric", "delta", "confidence", "power", "n_boot",
            "seed", "secondary_metrics", "labels_file", "label_source", "kind", "slot_types_file",
        }
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        for key in ("name", "metric", "delta"):
            if key not in cfg:
                raise ValueError(f"{path}: missing required key {key!r}")

        metrics = [cfg["metric"], *cfg.get("secondary_metrics", [])]
        for m in metrics:
            if m not in METRICS:
                raise ValueError(f"{path}: unknown metric {m!r}; choose from {sorted(METRICS)}")
        delta = float(cfg["delta"])
        if not 0 < delta < 1:
            raise ValueError(f"{path}: delta must be in (0, 1) (a fraction, e.g. 0.02 for 2 points)")

        kind = str(cfg.get("kind", "labels"))
        if kind not in ("labels", "frames"):
            raise ValueError(f"{path}: kind must be 'labels' or 'frames'")
        if kind == "frames":
            if cfg["metric"] != "accuracy" or cfg.get("secondary_metrics"):
                raise ValueError(f"{path}: frames tasks use metric 'accuracy' (exact frame match) and no secondary metrics")
            if "slot_types_file" not in cfg:
                raise ValueError(f"{path}: frames tasks need slot_types_file")

        def resolve(key: str) -> Path | None:
            value = cfg.get(key)
            if value is None:
                return None
            value = Path(value)
            return value if value.is_absolute() else (path.parent / value).resolve()

        labels_file = resolve("labels_file")

        return cls(
            name=str(cfg["name"]),
            metric=cfg["metric"],
            delta=delta,
            confidence=float(cfg.get("confidence", 0.95)),
            power=float(cfg.get("power", 0.8)),
            n_boot=int(cfg.get("n_boot", 10_000)),
            seed=int(cfg.get("seed", 0)),
            secondary_metrics=tuple(cfg.get("secondary_metrics", [])),
            labels_file=labels_file,
            label_source=str(cfg.get("label_source", "unspecified")),
            kind=kind,
            slot_types_file=resolve("slot_types_file"),
            path=path,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
