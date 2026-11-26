"""Kaggle kernel entry point (uploaded as the kernel's ``run.py``). Standard library only.

Installs Unsloth, unpacks the Mikan bundle from the attached private dataset, and runs
every SFT config in ``configs.json``. Kaggle's "GPU T4 x2" machine has two GPUs, so
configs are split into one queue per GPU and the queues run in parallel threads.
Outputs land in ``/kaggle/working/runs/<name>/`` and are downloaded with
``kaggle kernels output``.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

UNSLOTH = "unsloth==2026.11.12"
WORK = Path("/kaggle/working")
RUNS = WORK / "runs"


def find_bundle() -> Path:
    for pattern in ("/kaggle/input/**/bundle.tar.gz", "/kaggle/input/**/bundle.json"):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[0])
    raise SystemExit("Mikan bundle not found under /kaggle/input")


def unpack(src: Path) -> Path:
    """Copy or extract the bundle to a writable folder (``/kaggle/input`` is read-only, and
    Kaggle may have auto-extracted the archive there). ``/tmp`` keeps it out of the outputs."""
    root = Path("/tmp/mikan-bundle")
    shutil.rmtree(root, ignore_errors=True)
    if src.name == "bundle.json":
        shutil.copytree(src.parent, root)
    else:
        with tarfile.open(src) as tar:
            tar.extractall(root, filter="data")
    return root


def n_gpus() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return max(1, sum(line.startswith("GPU") for line in out.splitlines()))


def to_flags(cfg: dict) -> list[str]:
    flags = []
    for key, value in cfg.items():
        if key in ("name", "gpu"):
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value not in (False, None):
            flags += [flag, str(value)]
    return flags


def run_queue(gpu: int, queue: list[dict], root: Path, results: list[dict]) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(root), "HF_HOME": "/tmp/hf"}
    for cfg in queue:
        out = RUNS / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "mikan.train.sft", "--data", str(root / "data"), "--out", str(out.resolve()), *to_flags(cfg)]
        cwd = Path(f"/tmp/mikan-work-gpu{gpu}")  # libraries write caches relative to the working dir
        cwd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[gpu {gpu}] start {cfg['name']}", flush=True)
        with open(out / "log.txt", "w") as log:
            code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=cwd).returncode
        seconds = round(time.time() - t0)
        results.append({"name": cfg["name"], "gpu": gpu, "returncode": code, "seconds": seconds})
        tail = (out / "log.txt").read_text(errors="replace").splitlines()[-3:]
        print(f"[gpu {gpu}] done {cfg['name']} rc={code} in {seconds}s | " + " | ".join(tail), flush=True)


def main() -> None:
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", UNSLOTH], check=True)
    root = unpack(find_bundle())
    configs = json.loads((root / "configs.json").read_text())
    gpus = n_gpus()
    queues: dict[int, list[dict]] = {g: [] for g in range(gpus)}
    for i, cfg in enumerate(configs):
        queues[int(cfg.get("gpu", i)) % gpus].append(cfg)
    print(f"{len(configs)} configs on {gpus} GPU(s): " + json.dumps({g: [c['name'] for c in q] for g, q in queues.items()}), flush=True)

    results: list[dict] = []
    threads = [threading.Thread(target=run_queue, args=(g, q, root, results)) for g, q in queues.items() if q]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bundle_info = json.loads((root / "bundle.json").read_text())
    summary = {"bundle": bundle_info, "unsloth": UNSLOTH, "total_seconds": round(time.time() - t0), "runs": results}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
