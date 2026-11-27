"""Drive SFT runs on Kaggle GPUs from this machine.

    uv run python -m mikan.train.kaggle_job push --configs configs/sft_smoke.json --job mikan-sft-smoke
    uv run python -m mikan.train.kaggle_job status --job mikan-sft-smoke
    uv run python -m mikan.train.kaggle_job pull --job mikan-sft-smoke
    uv run python -m mikan.train.kaggle_job push --configs configs/teacher_massive_pilot.json \
        --job mikan-teacher-massive --runner teacher --data massive

``push`` bundles the ``mikan`` package, the prepared splits, the cached teacher labels (SFT
only) and the configs into one private Kaggle dataset (``mikan-bundle``), then pushes a
private GPU (T4 x2) script kernel that runs ``kaggle_runner.py`` (SFT) or
``mikan/intake/kaggle_teacher.py`` (``--runner teacher``). Nothing is public.

The Kaggle CLI is run through ``uvx`` by default; set ``MIKAN_KAGGLE_CMD`` to override.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET_SLUG = "mikan-bundle"
TEACHER_RUN = "runs/teacher-qwen3-30b-a3b-instruct-2507-fp8-10shot"
RUNNERS = {"sft": "mikan/train/kaggle_runner.py", "teacher": "mikan/intake/kaggle_teacher.py"}
DEFAULT_CMD = "uvx --from kaggle --with six kaggle"


def kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = shlex.split(os.environ.get("MIKAN_KAGGLE_CMD", DEFAULT_CMD)) + list(args)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def owner() -> str:
    out = kaggle("config", "view", capture=True).stdout
    for line in out.splitlines():
        if "username:" in line:
            return line.split("username:", 1)[1].strip()
    raise SystemExit("could not read the Kaggle username from `kaggle config view`")


def git_info() -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "mikan"))}


def build_bundle(configs: Path, out: Path, data: tuple[str, ...] = ("banking77",), with_teacher: bool = True) -> Path:
    """Write ``out/bundle.tar.gz`` with code, data, teacher labels, configs and provenance."""
    cfgs = json.loads(configs.read_text())
    entries = cfgs if isinstance(cfgs, list) else cfgs.get("runs", [])
    names = [c["name"] for c in entries]
    if len(set(names)) != len(names):
        raise SystemExit("config names must be unique")

    out.mkdir(parents=True, exist_ok=True)
    path = out / "bundle.tar.gz"
    info = {**git_info(), "configs": str(configs), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        ti.mtime = int(time.time())
        tar.addfile(ti, io.BytesIO(data))

    with tarfile.open(path, "w:gz") as tar:
        for py in sorted((REPO / "mikan").rglob("*.py")):
            tar.add(py, arcname=str(py.relative_to(REPO)))
        for d in data:
            for f in sorted((REPO / "data" / d).iterdir()):
                if f.is_file() and f.suffix in (".jsonl", ".txt", ".json"):
                    tar.add(f, arcname=f"data/{d}/{f.name}")
        if with_teacher:
            for split in ("train", "val", "golden"):
                tar.add(REPO / TEACHER_RUN / f"{split}_preds.jsonl", arcname=f"data/teacher/{split}_preds.jsonl")
        add_bytes(tar, "configs.json", json.dumps(cfgs, indent=2).encode())
        add_bytes(tar, "bundle.json", json.dumps(info, indent=2).encode())
    return path


def wait_dataset_ready(ref: str, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = kaggle("datasets", "status", ref, capture=True, check=False).stdout.strip().lower()
        if "ready" in status:
            return
        time.sleep(10)
    raise SystemExit(f"dataset {ref} not ready after {timeout}s")


def push(configs: Path, job: str, timeout_hours: float, runner: str = "sft", data: tuple[str, ...] = ("banking77",),
         dataset: str = DATASET_SLUG, with_teacher: bool | None = None) -> None:
    user = owner()
    build = REPO / "build" / "kaggle"
    ds_dir, k_dir = build / f"dataset-{dataset}", build / job
    for d in (ds_dir, k_dir):
        d.mkdir(parents=True, exist_ok=True)
    build_bundle(configs, ds_dir, data, with_teacher=(runner == "sft" and "banking77" in data) if with_teacher is None else with_teacher)
    ds_ref = f"{user}/{dataset}"
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(
        {"title": dataset, "id": ds_ref, "licenses": [{"name": "CC-BY-4.0"}]}, indent=2))

    exists = kaggle("datasets", "status", ds_ref, capture=True, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(ds_dir), "-m", f"bundle for {job}", "-q")
    else:
        kaggle("datasets", "create", "-p", str(ds_dir), "-q")  # private by default
    time.sleep(5)
    wait_dataset_ready(ds_ref)

    (k_dir / "run.py").write_text((REPO / RUNNERS[runner]).read_text())
    (k_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{job}",
        "title": job,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [ds_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    kaggle("kernels", "push", "-p", str(k_dir), "--accelerator", "NvidiaTeslaT4",
           "-t", str(int(timeout_hours * 3600)))
    print(f"pushed {user}/{job}; check with: uv run python -m mikan.train.kaggle_job status --job {job}")


def status(job: str) -> str:
    out = kaggle("kernels", "status", f"{owner()}/{job}", capture=True, check=False)
    text = (out.stdout + out.stderr).strip()
    print(text)
    return text


def pull(job: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{owner()}/{job}", "-p", str(dest), "-o", "-q")
    summary = dest / "summary.json"
    if summary.exists():
        info = json.loads(summary.read_text())
        if "chosen" in info:
            print(f"teacher chosen: {info['chosen']}")
        for r in info["runs"]:
            if "scores" in r:  # teacher run
                print(f"{r['name']:28s} {r['model']:55s} rc={r['returncode']} {r['seconds']:5d}s scores={r['scores']}")
                continue
            meta = dest / "runs" / r["name"] / "meta.json"
            best = json.loads(meta.read_text()) if meta.exists() else {}
            hist = {h["epoch"]: round(h["val_accuracy"], 4) for h in best.get("history", [])}
            print(f"{r['name']:40s} rc={r['returncode']} {r['seconds']:5d}s best_epoch={best.get('best_epoch')} val={hist}")
    print(f"downloaded to {dest}/")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run Mikan SFT jobs on Kaggle")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("push")
    a.add_argument("--configs", required=True, type=Path)
    a.add_argument("--job", required=True, help="kernel slug, e.g. mikan-sft-smoke")
    a.add_argument("--timeout-hours", type=float, default=9.0)
    a.add_argument("--runner", choices=sorted(RUNNERS), default="sft")
    a.add_argument("--data", default="banking77", help="comma list of data/ subdirs to bundle")
    a.add_argument("--dataset", default=DATASET_SLUG, help="private Kaggle dataset slug for the bundle")
    b = sub.add_parser("bundle", help="only build the bundle locally (for inspection)")
    b.add_argument("--configs", required=True, type=Path)
    s = sub.add_parser("status")
    s.add_argument("--job", required=True)
    o = sub.add_parser("pull")
    o.add_argument("--job", required=True)
    o.add_argument("--dest", type=Path)
    args = p.parse_args(argv)

    if args.cmd == "push":
        push(args.configs, args.job, args.timeout_hours, args.runner, tuple(x for x in args.data.split(",") if x), args.dataset)
    elif args.cmd == "bundle":
        print(build_bundle(args.configs, REPO / "build" / "kaggle" / "dataset"))
    elif args.cmd == "status":
        status(args.job)
    else:
        pull(args.job, args.dest or REPO / "runs" / "kaggle" / args.job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
