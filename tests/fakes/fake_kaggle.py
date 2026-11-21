"""Stand-in for the `kaggle` CLI so the Kaggle training flow can be tested offline.

Simulates: config view, datasets status/create/version, kernels push/status/output/logs.
Env: FAKE_KAGGLE_STATE (json state file), FAKE_KAGGLE_FAIL (training error), FAKE_KAGGLE_CRASH (kernel error).
"""
import json
import math
import os
import sys
import zipfile
from pathlib import Path

STATE = Path(os.environ["FAKE_KAGGLE_STATE"])
s = json.loads(STATE.read_text()) if STATE.exists() else {"datasets": {}, "kernels": {}, "calls": []}
args = sys.argv[1:]
s["calls"].append(args)
if os.environ.get("FAKE_KAGGLE_MISSING_MODULE"):  # like kaggle 2.2.5 without six installed
    print(f"ModuleNotFoundError: No module named '{os.environ['FAKE_KAGGLE_MISSING_MODULE']}'", file=sys.stderr)
    sys.exit(1)


def out(msg, code=0):
    print(msg)
    STATE.write_text(json.dumps(s))
    sys.exit(code)


def write_outputs(dest: Path, data_dir: Path):
    job = json.loads((data_dir / "job.json").read_text())
    n = len([ln for ln in (data_dir / "train.jsonl").read_text().splitlines() if ln.strip()])
    if os.environ.get("FAKE_KAGGLE_FAIL"):
        (dest / "metrics.json").write_text(json.dumps({
            "status": "error", "error": "OutOfMemoryError: CUDA out of memory",
            "traceback": "Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory"}))
        return
    q = job["quantization"].upper()
    name = f"{job['name']}.{q}.gguf"
    (dest / name).write_bytes(b"GGUF" + os.urandom(4096))
    # Like Unsloth's Qwen3-2507 Modelfile: a template but no PARAMETER lines.
    (dest / "Modelfile").write_text('FROM ./x.gguf\nTEMPLATE """{{ if .System }}<|im_start|>system\n'
                                    '{{ .System }}<|im_end|>{{ end }}"""\n')
    steps = math.ceil(math.ceil(n / job["batch_size"]) / job["grad_accum"]) * math.ceil(job["epochs"])
    (dest / "metrics.json").write_text(json.dumps({
        "status": "ok", "gpu": "Tesla T4", "examples": n, "train_runtime_s": 321.5, "final_loss": 0.8123,
        "steps": steps, "packing": job.get("packing", False),
        "labels": {"rows": n, "tokens": 40 * n, "trained_tokens": 10 * n, "rows_without_targets": 0,
                   "trained_preview": "a0<|im_end|>\n"},
        "gguf": name, "sample": {"prompt": "hi", "answer": "hello"}}))
    with zipfile.ZipFile(dest / "lora_adapter.zip", "w") as z:
        z.writestr("adapter_config.json", "{}")


if args[:2] == ["config", "view"]:
    out("Configuration values from /fake/.kaggle\n- username: fakeuser\n- auth_method: ACCESS_TOKEN")
if args[:2] == ["datasets", "status"]:
    if args[2] in s["datasets"]:
        out("ready")
    out("404 Client Error: Not Found", 1)
if args[:2] in (["datasets", "create"], ["datasets", "version"]):
    p = Path(args[args.index("-p") + 1])
    meta = json.loads((p / "dataset-metadata.json").read_text())
    if args[1] == "version" and meta["id"] not in s["datasets"]:
        out("404 Not Found", 1)
    s["datasets"][meta["id"]] = str(p)
    out(f"Dataset {args[1]} OK")
if args[:2] == ["kernels", "push"]:
    p = Path(args[args.index("-p") + 1])
    meta = json.loads((p / "kernel-metadata.json").read_text())
    if not (meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["machine_shape"]):
        out("Kernel push error: GPU/internet/machine_shape not set", 1)
    if not (p / meta["code_file"]).is_file():
        out("Kernel push error: code file missing", 1)
    ds = meta["dataset_sources"][0]
    if ds not in s["datasets"]:
        out(f"Kernel push error: dataset {ds} not found", 1)
    s["kernels"][meta["id"]] = {"polls": 0, "dataset": s["datasets"][ds], "meta": meta}
    out(f"Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/{meta['id']}")
if args[:2] == ["kernels", "status"]:
    k = s["kernels"][args[2]]
    k["polls"] += 1
    st = "QUEUED" if k["polls"] == 1 else "RUNNING" if k["polls"] == 2 else (
        "ERROR" if os.environ.get("FAKE_KAGGLE_CRASH") else "COMPLETE")
    out(f'{args[2]} has status "KernelWorkerStatus.{st}"')
if args[:2] == ["kernels", "output"]:
    dest = Path(args[args.index("-p") + 1])
    dest.mkdir(parents=True, exist_ok=True)
    if os.environ.get("FAKE_KAGGLE_CRASH"):
        out("No output files", 1)
    write_outputs(dest, Path(s["kernels"][args[2]]["dataset"]))
    out("Output files downloaded")
if args[:2] == ["kernels", "logs"]:
    out("step 1\nstep 2\nRuntimeError: CUDA error: device-side assert triggered")
out(f"fake kaggle: unsupported {args}", 2)
