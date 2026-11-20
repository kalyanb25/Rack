"""Look at this machine and say what it can realistically run."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Serving catalog: (approx. memory needed in GB incl. context, Ollama tag, note).
# Sizes are for the default Q4 builds with ~8K context. Check https://ollama.com/library for more.
SERVE_CATALOG = [
    (2.5, "qwen3:1.7b", "tiny, runs almost anywhere"),
    (4.0, "qwen3:4b-instruct-2507-q4_K_M", "best small all-rounder (default)"),
    (7.0, "qwen3:8b", "noticeably smarter, thinks before answering"),
    (11.0, "qwen3:14b", "strong general model"),
    (16.0, "gpt-oss:20b", "OpenAI's open-weight reasoning model"),
    (22.0, "qwen3:30b", "MoE: big-model quality, fast because only ~3B params are active"),
]


def _run(cmd: list[str], timeout: float = 10) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def ram_gb() -> float | None:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024 ** 2, 1)
        elif system == "Darwin":
            r = _run(["sysctl", "-n", "hw.memsize"])
            if r and r.returncode == 0:
                return round(int(r.stdout.strip()) / 1024 ** 3, 1)
        elif system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return round(stat.ullTotalPhys / 1024 ** 3, 1)
    except Exception:
        return None
    return None


def nvidia_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    r = _run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    if r and r.returncode == 0:
        for line in r.stdout.strip().splitlines():
            if "," not in line:
                continue
            name, mem = (x.strip() for x in line.rsplit(",", 1))
            try:
                gpus.append({"name": name, "vram_gb": round(int(float(mem)) / 1024, 1)})
            except ValueError:
                continue
    return gpus


def docker_status() -> dict:
    exe = shutil.which("docker")
    if not exe:
        return {"installed": False, "running": False, "compose": False}
    info = _run([exe, "info", "--format", "{{.ServerVersion}}"], timeout=20)
    compose = _run([exe, "compose", "version", "--short"])
    return {"installed": True,
            "running": bool(info and info.returncode == 0),
            "compose": bool(compose and compose.returncode == 0) or bool(shutil.which("docker-compose"))}


def kaggle_status() -> dict:
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    creds = bool(os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
                 or (cfg_dir / "access_token").exists() or (cfg_dir / "kaggle.json").exists())
    return {"cli": bool(shutil.which("kaggle")), "credentials": creds}


def unsloth_available(python: str) -> bool:
    r = _run([python, "-c", "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('unsloth') else 1)"],
             timeout=30)
    return bool(r and r.returncode == 0)


def detect(root: Path | None = None) -> dict:
    system = platform.system()
    machine = platform.machine().lower()
    disk = shutil.disk_usage(root or Path.cwd())
    return {
        "os": {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(system, system),
        "arch": machine,
        "python": platform.python_version(),
        "cpus": os.cpu_count() or 1,
        "ram_gb": ram_gb(),
        "disk_free_gb": round(disk.free / 1024 ** 3, 1),
        "gpus": nvidia_gpus(),
        "apple_silicon": system == "Darwin" and machine in ("arm64", "aarch64"),
        "wsl": "microsoft" in platform.release().lower(),
        "docker": docker_status(),
        "kaggle": kaggle_status(),
    }


def memory_budget(info: dict) -> tuple[float, str]:
    """How much memory a model can use here, and where it will run."""
    if info["gpus"]:
        return max(g["vram_gb"] for g in info["gpus"]), "NVIDIA GPU"
    ram = info["ram_gb"] or 4.0
    if info["apple_silicon"]:
        return round(ram * 0.65, 1), "Apple GPU (unified memory)"
    return round(ram * 0.6, 1), "CPU only (works, just slower)"


def recommend(info: dict) -> dict:
    budget, where = memory_budget(info)
    fits = [m for m in SERVE_CATALOG if m[0] <= budget]
    best = fits[-1] if fits else SERVE_CATALOG[0]
    if info["gpus"] and max(g["vram_gb"] for g in info["gpus"]) >= 8:
        if info["os"] == "Linux":
            train = ("local", "Your NVIDIA GPU can fine-tune small models. Install Unsloth in a venv "
                              "(`pip install unsloth`) and set local_train.python in rack.toml.")
        else:
            train = ("local-wsl", "Your NVIDIA GPU can fine-tune, easiest inside WSL2 (Ubuntu) with Unsloth. "
                                  "Or use Kaggle's free GPU with zero setup.")
    else:
        train = ("kaggle", "Train on Kaggle's free GPU (T4 x2, roughly 30 GPU-hours a week; check yours "
                           "with `kaggle quota`). rack packages, runs and downloads the job for you.")
    tips = []
    if info["os"] == "macOS":
        tips.append("On a Mac, install the native Ollama app (https://ollama.com/download) for GPU speed; "
                    "Docker on macOS can't use the Apple GPU. `rack up` will detect it and use it.")
    if not info["docker"]["installed"]:
        tips.append("Install Docker (Desktop on Mac/Windows, Engine on Linux) to run the web UI with `rack up`.")
    elif not info["docker"]["running"]:
        tips.append("Docker is installed but not running. Start it, then `rack up`.")
    if info["disk_free_gb"] < 20:
        tips.append(f"Only {info['disk_free_gb']} GB disk free. Each model version takes 1-10 GB.")
    if train[0] == "kaggle" and not (info["kaggle"]["cli"] and info["kaggle"]["credentials"]):
        tips.append("For free cloud training: `pip install kaggle`, then create an API token at "
                    "kaggle.com/settings and save it as ~/.kaggle/access_token (phone-verify your account "
                    "to unlock GPUs and internet).")
    return {"budget_gb": budget, "where": where, "serve": best, "fits": fits, "train": train, "tips": tips}


def python_ok() -> bool:
    return sys.version_info >= (3, 10)
