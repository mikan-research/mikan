"""Kaggle kernel entry point for teacher labelling on T4 x2 with vLLM (uploaded as ``run.py``).

Reads ``configs.json`` from the bundle:

    {"data": "massive",
     "candidates": [{"model": "...", "quantization": null}, ...],   # tried in order
     "probe": {"splits": "smoke", "shots": 1},
     "runs": [{"name": "massive-teacher-1shot", "splits": "pilot", "shots": 1},
              {"name": "massive-teacher-knn20", "splits": "pilot", "retrieved": 20}, ...]}

Each candidate runs the probe in its own process (a kernel crash only kills that process);
the first that produces valid output runs every entry in ``runs``. A failed candidate's
weights are deleted before the next is tried. Outputs: ``/kaggle/working/teacher/<name>/``
plus ``summary.json``. Standard library only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VLLM = "vllm==0.13.0"
WORK = Path("/kaggle/working")
HF_HOME = Path("/tmp/hf")


def find_bundle() -> Path:
    for p in Path("/kaggle/input").rglob("bundle.json"):
        return p.parent
    for p in Path("/kaggle/input").rglob("bundle.tar.gz"):
        return p
    raise SystemExit("bundle not found under /kaggle/input")


def unpack(src: Path) -> Path:
    dest = Path("/tmp/mikan-bundle")
    if dest.exists():
        shutil.rmtree(dest)
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        import tarfile

        dest.mkdir(parents=True)
        with tarfile.open(src) as tar:
            tar.extractall(dest)
    return dest


def run_predict(root: Path, cfg: dict, cand: dict, name: str, splits: str, shots: int, timeout: int, retrieved: int = 0) -> dict:
    out = WORK / "teacher" / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mikan.intake.vllm_predict", "--model", cand["model"],
           "--data", str(root / "data" / cfg["data"]), "--splits", splits, "--shots", str(shots),
           "--tp", str(cfg.get("tp", 2)), "--dtype", "float16", "--out", str(out),
           "--gpu-mem", str(cfg.get("gpu_mem", 0.90))]
    if cfg.get("attention_backend"):
        cmd += ["--attention-backend", cfg["attention_backend"]]
    if retrieved:
        cmd += ["--retrieved", str(retrieved)]
    if cand.get("quantization"):
        cmd += ["--quantization", cand["quantization"]]
    # FlashInfer JIT links against libcuda; Kaggle has only the driver's libcuda.so.1, not the
    # dev symlink, so point the linker at the toolkit stubs (Triton attention avoids the JIT anyway).
    lib = ":".join(x for x in ("/usr/local/cuda/lib64/stubs", os.environ.get("LIBRARY_PATH", "")) if x)
    env = {**os.environ, "PYTHONPATH": str(root), "HF_HOME": str(HF_HOME), "LIBRARY_PATH": lib,
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"}
    t0 = time.time()
    with open(out / "log.txt", "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        try:
            code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd="/tmp",
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = -999
    scores = {}
    for s in splits.split(","):
        meta = out / f"{s.strip()}_meta.json"
        if meta.exists():
            scores[s.strip()] = json.loads(meta.read_text())["accuracy_vs_gold"]
    return {"name": name, "model": cand["model"], "returncode": code, "seconds": int(time.time() - t0), "scores": scores}


def main() -> None:
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", VLLM], check=True)
    root = unpack(find_bundle())
    cfg = json.loads((root / "configs.json").read_text())
    results: list[dict] = []
    chosen = None
    for i, cand in enumerate(cfg["candidates"]):
        probe = cfg["probe"]
        r = run_predict(root, cfg, cand, f"probe-{i}", probe["splits"], probe.get("shots", 0),
                        cfg.get("probe_timeout", 2400), probe.get("retrieved", 0))
        r["probe"] = True
        results.append(r)
        print(json.dumps(r), flush=True)
        if r["returncode"] == 0 and r["scores"]:
            chosen = cand
            break
        shutil.rmtree(HF_HOME, ignore_errors=True)  # free disk before the next candidate
    if chosen:
        for run in cfg["runs"]:
            r = run_predict(root, cfg, chosen, run["name"], run["splits"], run.get("shots", 0),
                            cfg.get("run_timeout", 10 * 3600), run.get("retrieved", 0))
            results.append(r)
            print(json.dumps(r), flush=True)
    (WORK / "summary.json").write_text(json.dumps({"chosen": chosen, "runs": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
