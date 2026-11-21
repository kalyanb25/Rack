"""Run a training job on Kaggle's free GPU through the official Kaggle CLI.

Flow: upload data as a private dataset -> push a private GPU script kernel that uses it ->
poll until it finishes -> download the outputs (GGUF, Modelfile, metrics, LoRA adapter).
CLI docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .. import ui
from ..config import RackError


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


class Kaggle:
    def __init__(self, cfg: dict):
        self.cfg = cfg["kaggle"]
        exe = shutil.which("kaggle")
        if not exe:
            raise RackError("Kaggle CLI not found. Install it with `pip install kaggle six`, then put an API token "
                            "from kaggle.com/settings in ~/.kaggle/access_token (or set KAGGLE_API_TOKEN).")
        self.exe = exe
        self.username = self._username()

    def run(self, *args: str, check: bool = True, timeout: float = 900) -> str:
        try:
            r = subprocess.run([self.exe, *args], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RackError(f"`kaggle {' '.join(args)}` timed out after {int(timeout)}s") from e
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        missing = re.search(r"No module named '([\w.]+)'", out)
        if missing:
            pkg = missing.group(1).split(".")[0]
            raise RackError(f"The Kaggle CLI is missing a Python package it needs ({pkg}). "
                            f"Fix: pip install {pkg}")
        if check and r.returncode != 0:
            raise RackError(f"`kaggle {' '.join(args)}` failed:\n{out[-2000:]}")
        return out

    def _username(self) -> str:
        if self.cfg.get("username"):
            return self.cfg["username"]
        cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        legacy = cfg_dir / "kaggle.json"
        if legacy.is_file():
            try:
                u = json.loads(legacy.read_text()).get("username")
                if u:
                    return u
            except (OSError, json.JSONDecodeError):
                pass
        out = self.run("config", "view", check=False, timeout=60)
        m = re.search(r"username:\s*(\S+)", out)
        if m and m.group(1).lower() not in ("none", "null", "''"):
            return m.group(1)
        raise RackError("Couldn't work out your Kaggle username. Set [kaggle] username in rack.toml "
                        "or the KAGGLE_USERNAME environment variable.")

    # ---- dataset (your training data, private) ---------------------------
    def dataset_status(self, ref: str) -> str | None:
        r = subprocess.run([self.exe, "datasets", "status", ref], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        return lines[-1].lower() if lines else None

    def upload_dataset(self, data_dir: Path, slug: str, note: str) -> str:
        ref = f"{self.username}/{slug}"
        meta = {"title": slug, "id": ref, "licenses": [{"name": "other"}],
                "description": "Private fine-tuning data uploaded by rack. Not for redistribution."}
        (data_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self.dataset_status(ref) is None:
            ui.step(f"Creating private Kaggle dataset {ref}")
            self.run("datasets", "create", "-p", str(data_dir), "-q", "-t")  # -t: keep JSON files as-is
        else:
            ui.step(f"Uploading a new version of {ref}")
            self.run("datasets", "version", "-p", str(data_dir), "-m", note, "-q", "-t", "-d")
            time.sleep(float(self.cfg.get("dataset_settle_seconds", 20)))  # let processing start
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            status = self.dataset_status(ref) or ""
            if "ready" in status:
                ui.ok("Dataset ready")
                return ref
            if "error" in status:
                raise RackError(f"Kaggle couldn't process the dataset {ref} (status: {status}).")
            time.sleep(10)
        raise RackError(f"Dataset {ref} wasn't ready after 15 minutes. Check it on kaggle.com, then retry.")

    # ---- kernel (the GPU job) --------------------------------------------
    def write_kernel(self, kernel_dir: Path, slug: str, dataset_ref: str, script: Path,
                     job_sha256: str | None = None) -> str:
        ref = f"{self.username}/{slug}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        code = script.read_text(encoding="utf-8")
        if job_sha256:
            marker = "EXPECTED_JOB_SHA256 = None"
            if marker not in code:
                raise RackError("train_unsloth.py is missing its EXPECTED_JOB_SHA256 marker.")
            code = code.replace(marker, f'EXPECTED_JOB_SHA256 = "{job_sha256}"', 1)
        (kernel_dir / script.name).write_text(code, encoding="utf-8")
        meta = {
            "id": ref,
            "title": slug,
            "code_file": script.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": self.cfg.get("accelerator") or "NvidiaTeslaT4",
            "dataset_sources": [dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return ref

    def push(self, kernel_dir: Path) -> None:
        out = self.run("kernels", "push", "-p", str(kernel_dir), timeout=600)
        if "error" in out.lower() and "successfully" not in out.lower():
            raise RackError(f"Kaggle rejected the job:\n{out[-1500:]}")

    def status(self, ref: str) -> str:
        """complete | error | cancel | running | queued, or missing/unknown when Kaggle gave no status."""
        out = self.run("kernels", "status", ref, check=False, timeout=120)
        m = re.search(r'has status "([^"]+)"', out)
        if not m:  # an API error, not the job's status (its text may well contain the word "error")
            self.last_status_output = out
            low = out.lower()
            return "missing" if ("404" in low or "not found" in low) else "unknown"
        raw = m.group(1).lower()
        for key in ("complete", "error", "cancel", "running", "queued"):
            if key in raw:
                return key
        return raw.strip() or "unknown"

    def wait(self, ref: str, timeout_min: float, poll_s: float) -> str:
        t0 = time.monotonic()
        last = ""
        unanswered = 0
        while True:
            s = self.status(ref)
            unanswered = unanswered + 1 if s in ("missing", "unknown") else 0
            if unanswered >= 10:
                raise RackError(f"Kaggle stopped answering about {ref}. Last reply:\n"
                                f"{getattr(self, 'last_status_output', '')[-800:]}")
            mins = int((time.monotonic() - t0) / 60)
            if s != last:
                ui.info(f"  [{mins:3d} min] {s}")
                last = s
            if s in ("complete", "error", "cancel"):
                return s
            if time.monotonic() - t0 > timeout_min * 60:
                raise RackError(f"Training still not done after {int(timeout_min)} min. It keeps running on "
                                f"Kaggle; later run `rack train --resume <job folder>` to collect it.")
            time.sleep(poll_s)

    def download(self, ref: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.run("kernels", "output", ref, "-p", str(out_dir), "-o", timeout=3600)

    def logs_tail(self, ref: str, lines: int = 40) -> str:
        out = self.run("kernels", "logs", ref, check=False, timeout=180)
        return "\n".join(out.splitlines()[-lines:])


def kernel_slug(project_name: str) -> str:
    return _slug(f"rack-train-{project_name}")[:50]


def dataset_slug(project_name: str) -> str:
    return _slug(f"rack-data-{project_name}")[:50]
