#!/usr/bin/env python3
"""rack fine-tuning job: LoRA fine-tune an open-weight model with Unsloth, export GGUF for Ollama.

Runs anywhere with an NVIDIA GPU:
  * Kaggle (free T4s): `rack train <model>` uploads and runs this for you.
  * Your own GPU:      `rack train <model> --backend local`
  * Colab or anywhere: python train_unsloth.py --job data/job.json --out output/

Inputs:  job.json + train.jsonl (+ optional valid.jsonl) in the same folder.
Outputs: <name>.<QUANT>.gguf, Modelfile (chat template for Ollama), lora_adapter.zip, metrics.json

Standalone on purpose: it must not import anything from rack.
The training recipe mirrors Unsloth's official Qwen3-4B-Instruct notebook.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()

# Kaggle's free tier gives two T4s. Unsloth's single-GPU path is the reliable one, so use one.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Unsloth's tested install recipe for Kaggle (from their Kaggle-Qwen3_(4B)-Instruct notebook).
# If Unsloth changes it, override with "install": [...] in job.json (rack: [train] install = [...]).
# If xformers has no wheel for this Python, the first line is retried without it (Unsloth runs fine
# without xformers).
KAGGLE_INSTALL = [
    "pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128",
    "pip install -q unsloth",
    'pip install -q --no-deps --upgrade "torchao>=0.16.0"',
    "pip install -q transformers==4.57.6",
    "pip install -q --no-deps trl==0.22.2",
]

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# rack fills this in for Kaggle runs: the sha256 of the exact job.json this run must use. Kaggle can
# attach the previous version of a dataset that is still processing, so this guards against training
# on stale data.
EXPECTED_JOB_SHA256 = None


def log(msg: str) -> None:
    print(f"[rack {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_job(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_file():
            raise SystemExit(f"job file not found: {p}")
        return p
    roots = [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.is_dir():
            hits = sorted(root.rglob("job.json"))
            if hits:
                return hits[0]
    raise SystemExit("job.json not found. Pass --job path/to/job.json "
                     "(on Kaggle, attach the rack data dataset to this notebook).")


def pick_chat_template(base: str, configured: str | None) -> str | None:
    """Unsloth chat template name for common bases. None = keep the tokenizer's own template."""
    if configured and configured != "auto":
        return configured
    b = base.lower()
    if "qwen3" in b and "instruct-2507" in b:
        return "qwen3-instruct"
    if "qwen3" in b and "thinking-2507" in b:
        return "qwen3-thinking"
    if "qwen3-" in b:
        return "qwen3"
    if "qwen2.5" in b:
        return "qwen-2.5"
    if "llama-3" in b:
        return "llama-3.1"
    if "gemma-4" in b:
        return "gemma-4"
    if "gemma-3" in b:
        return "gemma-3"
    if "phi-4" in b:
        return "phi-4"
    return None


def pick_gguf(files: list, quant: str) -> Path | None:
    paths = [Path(f) for f in files if str(f).lower().endswith(".gguf") and Path(f).is_file()]
    q = quant.lower().replace("-", "_")
    matching = [p for p in paths if q in p.name.lower()]
    candidates = matching or paths
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.stat().st_size)  # quantized file is the smallest one


def sh(cmd: str) -> None:
    """Run a shell command; on failure raise with its real output so metrics.json explains it."""
    log(f"$ {cmd}")
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1",  # Kaggle's Python is a throwaway system Python
               PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_ROOT_USER_ACTION="ignore")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    tail = "\n".join(((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-25:])
    if tail:
        print(tail, flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"`{cmd}` failed (exit {r.returncode}):\n{tail}")


def preflight() -> None:
    """Fail in seconds, with a clear reason, instead of minutes into an install."""
    import socket
    try:
        socket.create_connection(("pypi.org", 443), timeout=15).close()
    except OSError as e:
        raise SystemExit("This Kaggle session has no internet, so it can't install Unsloth or download the base "
                         "model. Kaggle only allows internet for phone-verified accounts: kaggle.com > Settings > "
                         f"Phone verification. Then run the training again. ({e})")
    smi = shutil.which("nvidia-smi")
    gpus = subprocess.run([smi, "-L"], capture_output=True, text=True).stdout.strip() if smi else ""
    if not gpus:
        raise SystemExit("This Kaggle session has no GPU. Kaggle gives GPUs only to phone-verified accounts with "
                         "weekly GPU quota left (see kaggle.com > Settings, or `kaggle quota`).")
    log("GPUs: " + gpus.replace("\n", "; "))


def install_if_needed(job: dict) -> None:
    if importlib.util.find_spec("unsloth") is not None:
        return
    if not (ON_KAGGLE or job.get("auto_install")):
        raise SystemExit("Unsloth isn't installed in this Python. Install it first "
                         "(see https://unsloth.ai/docs) or let rack run this on Kaggle.")
    log(f"installing Unsloth for Python {sys.version.split()[0]} (takes a few minutes)...")
    for cmd in job.get("install") or KAGGLE_INSTALL:
        if cmd.startswith("pip "):  # use this interpreter's pip, whatever `pip` on PATH is
            cmd = f'"{sys.executable}" -m {cmd}'
        try:
            sh(cmd)
        except RuntimeError:
            if " xformers" not in cmd:
                raise
            log("xformers didn't install for this Python; continuing without it")
            sh(cmd.replace(" xformers", ""))
    importlib.invalidate_caches()


def zip_dir(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(src))


def train(job: dict, data_dir: Path, out_dir: Path, work_dir: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible. On Kaggle, the notebook needs a GPU accelerator; "
                           "locally, check that nvidia-smi works.")
    gpu = torch.cuda.get_device_name(0)
    log(f"GPU: {gpu} ({round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB)")

    from unsloth import FastLanguageModel  # import before trl/transformers so Unsloth can patch them
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    log(f"loading {job['base']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=job["base"],
        max_seq_length=int(job["max_seq_length"]),
        load_in_4bit=bool(job.get("load_in_4bit", True)),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(job["lora_r"]),
        target_modules=LORA_TARGETS,
        lora_alpha=int(job["lora_alpha"]),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    template = pick_chat_template(job["base"], job.get("chat_template"))
    if template:
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=template)
            log(f"chat template: {template}")
        except Exception as e:  # unknown name in this Unsloth version
            log(f"keeping the tokenizer's own chat template ({e})")

    def to_text(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                         for m in batch["messages"]]}

    # Keep only the rendered "text" column so the trainer can't re-template the raw messages.
    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    train_ds = train_ds.map(to_text, batched=True, remove_columns=train_ds.column_names)
    eval_ds = None
    if (data_dir / "valid.jsonl").is_file():
        eval_ds = load_dataset("json", data_files=str(data_dir / "valid.jsonl"), split="train")
        eval_ds = eval_ds.map(to_text, batched=True, remove_columns=eval_ds.column_names)
    log(f"{len(train_ds)} training examples" + (f", {len(eval_ds)} validation" if eval_ds else ""))

    cfg = dict(
        dataset_text_field="text",
        per_device_train_batch_size=int(job["batch_size"]),
        gradient_accumulation_steps=int(job["grad_accum"]),
        warmup_steps=5,
        learning_rate=float(job["learning_rate"]),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        output_dir=str(work_dir / "checkpoints"),
        save_strategy="no",
    )
    if int(job.get("max_steps") or 0) > 0:
        cfg["max_steps"] = int(job["max_steps"])
    else:
        cfg["num_train_epochs"] = float(job["epochs"])
    if eval_ds is not None:
        cfg["eval_strategy"] = "epoch"

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds,
                         eval_dataset=eval_ds, args=SFTConfig(**cfg))
    try:
        trainer = train_on_responses_only(trainer)
        log("loss on assistant replies only")
    except Exception as e:
        log(f"loss on full conversations (couldn't isolate replies: {e})")

    t0 = time.time()
    stats = trainer.train()
    runtime = round(time.time() - t0, 1)
    history = [{k: h[k] for k in ("step", "epoch", "loss", "eval_loss") if k in h}
               for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    losses = [h["loss"] for h in history if "loss" in h]
    log(f"trained in {runtime}s, final loss {losses[-1] if losses else 'n/a'}")

    sample = None
    try:
        first = json.loads((data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        prompt_msgs = [m for m in first["messages"] if m["role"] != "assistant"][:2]
        FastLanguageModel.for_inference(model)
        text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20)
        answer = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        sample = {"prompt": prompt_msgs[-1]["content"], "answer": answer.strip()}
        log(f"sample answer: {answer.strip()[:200]}")
    except Exception as e:
        log(f"skipped sample generation ({e})")

    lora_dir = work_dir / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    zip_dir(lora_dir, out_dir / "lora_adapter.zip")
    log("saved LoRA adapter")

    quant = job.get("quantization", "q4_k_m")
    log(f"exporting GGUF ({quant}); this builds llama.cpp and takes 5-10 minutes")
    res = model.save_pretrained_gguf(str(work_dir / "model"), tokenizer, quantization_method=quant,
                                     temporary_location=str(work_dir / "_unsloth_tmp"))
    res = res if isinstance(res, dict) else {}
    files = res.get("gguf_files") or glob.glob(str(work_dir / "model_gguf" / "*.gguf")) \
        or glob.glob(str(work_dir / "**" / "*.gguf"), recursive=True)
    gguf = pick_gguf(files, quant)
    if gguf is None:
        raise RuntimeError("GGUF export produced no .gguf file")
    final = out_dir / f"{job['name']}.{quant.upper()}.gguf"
    shutil.move(str(gguf), final)
    modelfile = Path(res.get("modelfile_location") or (work_dir / "model_gguf" / "Modelfile"))
    if modelfile.is_file():
        shutil.copy2(modelfile, out_dir / "Modelfile")
    log(f"wrote {final.name} ({round(final.stat().st_size / 1024**3, 2)} GB)")

    import transformers
    import trl
    import unsloth
    return {
        "gpu": gpu,
        "examples": len(train_ds),
        "train_runtime_s": runtime,
        "train_loss": getattr(stats, "training_loss", None),
        "final_loss": losses[-1] if losses else None,
        "loss_history": history[-200:],
        "chat_template": template,
        "gguf": final.name,
        "quantization": quant,
        "modelfile": (out_dir / "Modelfile").is_file(),
        "sample": sample,
        "versions": {"unsloth": getattr(unsloth, "__version__", "?"),
                     "transformers": transformers.__version__, "trl": trl.__version__,
                     "torch": torch.__version__},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", help="path to job.json (auto-detected on Kaggle)")
    ap.add_argument("--out", help="output folder (default: /kaggle/working on Kaggle, else ./output)")
    args = ap.parse_args()

    job_path = find_job(args.job).resolve()
    job_bytes = job_path.read_bytes()
    job = json.loads(job_bytes)
    data_dir = job_path.parent
    out_dir = Path(args.out) if args.out else (Path("/kaggle/working") if ON_KAGGLE else Path.cwd() / "output")
    work_dir = Path("/tmp/rack-work") if ON_KAGGLE else out_dir.resolve() / "_work"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Unsloth drops caches (e.g. unsloth_compiled_cache/) in the working directory. Keep them out of
    # the output folder, which on Kaggle is exactly what gets downloaded.
    os.chdir(work_dir)

    metrics: dict = {"status": "running", "name": job.get("name"), "base": job.get("base"),
                     "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "where": "kaggle" if ON_KAGGLE else "local"}
    try:
        if EXPECTED_JOB_SHA256 and hashlib.sha256(job_bytes).hexdigest() != EXPECTED_JOB_SHA256:
            raise SystemExit("Kaggle attached an older version of your training data (the new upload was "
                             "still processing). Nothing was trained. Run `rack train` again in a minute.")
        if ON_KAGGLE:
            preflight()
        install_if_needed(job)
        metrics.update(train(job, data_dir, out_dir, work_dir))
        metrics["status"] = "ok"
    except BaseException as e:  # record everything, including SystemExit from setup
        metrics["status"] = "error"
        metrics["error"] = f"{type(e).__name__}: {e}"
        metrics["traceback"] = traceback.format_exc()[-6000:]
        log("FAILED: " + metrics["error"])
    finally:
        metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        os.chdir(out_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

    # On Kaggle, exit cleanly even on failure so the output (metrics.json with the error) is kept
    # and rack can show you what went wrong.
    if metrics["status"] != "ok" and not ON_KAGGLE:
        sys.exit(1)


if __name__ == "__main__":
    main()
