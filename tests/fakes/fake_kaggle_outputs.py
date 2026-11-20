import json
import os
import zipfile
from pathlib import Path


def write(dest: Path, data_dir: Path):
    job = json.loads((data_dir / "job.json").read_text())
    n = len([ln for ln in (data_dir / "train.jsonl").read_text().splitlines() if ln.strip()])
    q = job["quantization"].upper()
    name = f"{job['name']}.{q}.gguf"
    (dest / name).write_bytes(b"GGUF" + os.urandom(2048))
    (dest / "metrics.json").write_text(json.dumps({
        "status": "ok", "gpu": "RTX 4090", "examples": n, "train_runtime_s": 99.0, "final_loss": 0.5,
        "gguf": name}))
    with zipfile.ZipFile(dest / "lora_adapter.zip", "w") as z:
        z.writestr("adapter_config.json", "{}")
