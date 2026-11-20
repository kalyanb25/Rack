"""Prepare, run and collect fine-tuning jobs (local GPU or Kaggle's free GPU)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .. import __version__, data, ops, ui
from ..config import RackError
from ..doctor import nvidia_gpus, unsloth_available
from ..ollama import Ollama
from ..project import Project
from .kaggle import Kaggle, dataset_slug, kernel_slug

SCRIPT = Path(__file__).parent / "job" / "train_unsloth.py"


def prepare_job(root: Path, project: Project) -> Path:
    t = project.cfg["train"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = root / ".rack" / "jobs" / f"{project.name}-{stamp}"
    n = 2
    while job_dir.exists():
        job_dir = root / ".rack" / "jobs" / f"{project.name}-{stamp}-{n}"
        n += 1
    data_dir = job_dir / "data"
    system = project.system_prompt() if t.get("include_system", True) else None
    try:
        stats = data.prepare(project.path(t["data"]), data_dir / "train.jsonl", system)
        if t.get("valid"):
            data.prepare(project.path(t["valid"]), data_dir / "valid.jsonl", system)
    except ValueError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise RackError(str(e)) from e
    job = {
        "name": project.name,
        "base": project.base_hf,
        "max_seq_length": t["max_seq_length"],
        "lora_r": t["lora_r"],
        "lora_alpha": t.get("lora_alpha", t["lora_r"]),
        "epochs": t["epochs"],
        "max_steps": t.get("max_steps", 0),
        "learning_rate": t["learning_rate"],
        "batch_size": t["batch_size"],
        "grad_accum": t["grad_accum"],
        "quantization": t["quantization"],
        "chat_template": t.get("chat_template", "auto"),
        "load_in_4bit": t.get("load_in_4bit", True),
        "examples": stats["count"],
        "data_sha256": stats["sha256"],
        "rack_version": __version__,
    }
    if t.get("install"):
        job["install"] = list(t["install"])
    job_bytes = json.dumps(job, indent=2).encode("utf-8")
    (data_dir / "job.json").write_bytes(job_bytes)
    shutil.copy2(SCRIPT, job_dir / SCRIPT.name)
    _save_meta(job_dir, {"project": project.name, "created": stamp, "status": "prepared",
                         "job_sha256": hashlib.sha256(job_bytes).hexdigest()})
    return job_dir


def _meta_path(job_dir: Path) -> Path:
    return job_dir / "rack-job.json"


def _load_meta(job_dir: Path) -> dict:
    p = _meta_path(job_dir)
    if not p.is_file():
        raise RackError(f"{job_dir} isn't a rack training job folder.")
    return json.loads(p.read_text(encoding="utf-8"))


def _save_meta(job_dir: Path, meta: dict) -> None:
    _meta_path(job_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def resolve_backend(requested: str, cfg: dict) -> str:
    if requested in ("local", "kaggle"):
        return requested
    if requested != "auto":
        raise RackError(f"Unknown backend {requested!r}: use auto, kaggle or local.")
    if nvidia_gpus() and unsloth_available(cfg["local_train"]["python"]):
        return "local"
    if shutil.which("kaggle"):
        return "kaggle"
    raise RackError("No way to train yet. Easiest free option: `pip install kaggle`, create an API token at "
                    "kaggle.com/settings, save it as ~/.kaggle/access_token, then run this again. "
                    "Got an NVIDIA GPU? Install Unsloth and use --backend local.")


def run_local(job_dir: Path, cfg: dict) -> None:
    py = cfg["local_train"]["python"]
    meta = _load_meta(job_dir)
    meta.update({"backend": "local", "status": "running"})
    _save_meta(job_dir, meta)
    ui.step(f"Training locally with {py} (Ctrl+C stops it)")
    r = subprocess.run([py, str(job_dir / SCRIPT.name), "--job", str(job_dir / "data" / "job.json"),
                        "--out", str(job_dir / "output")])
    if r.returncode != 0 and not (job_dir / "output" / "metrics.json").is_file():
        raise RackError(f"Training process exited with code {r.returncode}.")


def run_kaggle(job_dir: Path, project: Project, cfg: dict, note: str) -> None:
    k = Kaggle(cfg)
    meta = _load_meta(job_dir)
    ds = k.upload_dataset(job_dir / "data", dataset_slug(project.name), note)
    ref = k.write_kernel(job_dir / "kernel", kernel_slug(project.name), ds, job_dir / SCRIPT.name,
                         job_sha256=meta.get("job_sha256"))
    ui.step(f"Starting the GPU job on Kaggle ({cfg['kaggle'].get('accelerator')})")
    k.push(job_dir / "kernel")
    meta.update({"backend": "kaggle", "kernel": ref, "dataset": ds, "status": "running"})
    _save_meta(job_dir, meta)
    ui.info(f"  Watch it live: https://www.kaggle.com/code/{ref}")
    ui.info(ui.dim("  Usually 20-40 min for a 4B model (install + train + GGUF export). "
                   "Safe to Ctrl+C: resume later with `rack train --resume " + str(job_dir) + "`"))
    _wait_and_download(k, job_dir, cfg)


def _wait_and_download(k: Kaggle, job_dir: Path, cfg: dict) -> None:
    meta = _load_meta(job_dir)
    ref = meta["kernel"]
    status = k.wait(ref, float(cfg["kaggle"]["timeout_minutes"]), float(cfg["kaggle"]["poll_seconds"]))
    ui.step("Downloading results from Kaggle")
    try:
        k.download(ref, job_dir / "output")
    except RackError as e:
        if status == "complete":
            raise
        ui.warn(f"Couldn't download outputs: {e}")
    meta["status"] = status
    _save_meta(job_dir, meta)
    if status != "complete" and not (job_dir / "output" / "metrics.json").is_file():
        tail = k.logs_tail(ref)
        raise RackError(f"The Kaggle job ended with status '{status}'. Last log lines:\n{tail}\n"
                        f"Full logs: https://www.kaggle.com/code/{ref}")


def collect(client: Ollama, job_dir: Path, project: Project, notes: str = "") -> dict:
    out = job_dir / "output"
    mpath = out / "metrics.json"
    if not mpath.is_file():
        raise RackError(f"No results in {out}. If the job is still running, resume it later.")
    metrics = json.loads(mpath.read_text(encoding="utf-8"))
    if metrics.get("status") != "ok":
        tb = "\n".join((metrics.get("traceback") or "").strip().splitlines()[-12:])
        raise RackError(f"Training failed: {metrics.get('error')}\n{tb}\n"
                        "Common fixes: smaller max_seq_length or batch_size (out of memory), "
                        "or check the base model name in model.toml.")
    job = json.loads((job_dir / "data" / "job.json").read_text(encoding="utf-8"))
    gguf = out / metrics["gguf"] if metrics.get("gguf") else None
    if not gguf or not gguf.is_file():
        found = sorted(out.glob("*.gguf"))
        if not found:
            raise RackError(f"Training finished but no .gguf file is in {out}.")
        gguf = found[0]
    modelfile = out / "Modelfile"
    meta = _load_meta(job_dir)
    train_meta = {
        "backend": meta.get("backend", "local"), "job": job_dir.name,
        "examples": metrics.get("examples"), "final_loss": metrics.get("final_loss"),
        "train_runtime_s": metrics.get("train_runtime_s"), "gpu": metrics.get("gpu"),
        "data_sha256": job["data_sha256"][:16], "epochs": job["epochs"],
        "learning_rate": job["learning_rate"], "lora_r": job["lora_r"],
        "quantization": job["quantization"], "sample": metrics.get("sample"),
    }
    loss = metrics.get("final_loss")
    ui.ok(f"Training done on {metrics.get('gpu')}: {metrics.get('examples')} examples, "
          f"{metrics.get('train_runtime_s')}s, final loss {round(loss, 4) if isinstance(loss, float) else loss}")
    entry = ops.import_gguf(client, project, gguf, modelfile=modelfile if modelfile.is_file() else None,
                            kind="finetune", notes=notes, train=train_meta, base=job["base"])
    meta["status"] = "imported"
    meta["version"] = entry["version"]
    _save_meta(job_dir, meta)
    return entry


def resume(client: Ollama, root: Path, job_dir: Path, cfg: dict, notes: str = "") -> dict:
    job_dir = job_dir if job_dir.is_absolute() else (Path.cwd() / job_dir)
    meta = _load_meta(job_dir)
    project = Project(root, meta["project"])
    if meta.get("status") == "imported":
        raise RackError(f"That job was already imported as {project.name}:{meta.get('version')}.")
    if meta.get("backend") == "kaggle" and not (job_dir / "output" / "metrics.json").is_file():
        _wait_and_download(Kaggle(cfg), job_dir, cfg)
    return collect(client, job_dir, project, notes)
