"""Start/stop the Docker stack (Ollama + Open WebUI + optional API gateway)."""

from __future__ import annotations

import os
import platform
import secrets
import shutil
import subprocess
import time
from pathlib import Path

from . import ui
from .config import RackError
from .doctor import nvidia_gpus
from .ollama import Ollama, OllamaError


def compose_cmd() -> list[str]:
    docker = shutil.which("docker")
    if docker:
        r = subprocess.run([docker, "compose", "version"], capture_output=True, text=True)
        if r.returncode == 0:
            return [docker, "compose"]
    legacy = shutil.which("docker-compose")
    if legacy:
        return [legacy]
    raise RackError("Docker Compose not found. Install Docker Desktop (Mac/Windows) or Docker Engine + the "
                    "compose plugin (Linux): https://docs.docker.com/get-docker/")


def read_env(root: Path) -> dict:
    env = {}
    p = root / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def set_env_value(root: Path, key: str, value: str) -> None:
    p = root / ".env"
    lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_env(root: Path) -> dict:
    """Create .env from .env.example the first time, with fresh secrets."""
    env_file, example = root / ".env", root / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copy2(example, env_file)
        ui.ok("Created .env from .env.example")
    env = read_env(root)
    if not env.get("WEBUI_SECRET_KEY"):
        set_env_value(root, "WEBUI_SECRET_KEY", secrets.token_urlsafe(32))
    return read_env(root)


def running_services(root: Path) -> set[str]:
    try:
        r = subprocess.run(compose_cmd() + ["ps", "--status", "running", "--services"], cwd=root,
                           capture_output=True, text=True, timeout=30)
    except (RackError, subprocess.SubprocessError, OSError):
        return set()
    return set(r.stdout.split()) if r.returncode == 0 else set()


def up(root: Path, *, gpu: str = "auto", gateway: bool = False, native: str = "auto") -> dict:
    base = compose_cmd()
    from .doctor import docker_status
    if not docker_status()["running"]:
        raise RackError("Docker isn't running. Start Docker Desktop (or `sudo systemctl start docker` on Linux), "
                        "then run `rack up` again.")
    env = ensure_env(root)
    local = Ollama("http://localhost:11434", timeout=5)
    try:
        local.version()
        reachable = True
    except OllamaError:
        reachable = False
    ours = "ollama" in running_services(root)
    use_native = native == "yes" or (native == "auto" and reachable and not ours)
    if native == "yes" and not reachable:
        raise RackError("--native: nothing is answering on localhost:11434. Start the Ollama app first.")

    run_env = os.environ.copy()
    files = ["-f", "docker-compose.yml"]
    services: list[str] = []
    gpu_on = False
    if use_native:
        ui.ok("Found Ollama already running on this machine; the web UI will use it")
        if platform.system() == "Linux":
            ui.warn("On Linux the Ollama service listens on 127.0.0.1 only, so the web UI container may not "
                    "reach it. Either set OLLAMA_HOST=0.0.0.0 for it (`sudo systemctl edit ollama`), or stop it "
                    "(`sudo systemctl disable --now ollama`) and run `rack up` again to use Docker's Ollama.")
        run_env["OLLAMA_BASE_URL"] = "http://host.docker.internal:11434"
        run_env["OLLAMA_UPSTREAM"] = "host.docker.internal:11434"
        services = ["--no-deps", "open-webui"] + (["gateway"] if gateway else [])
    else:
        if platform.system() == "Darwin":
            ui.warn("Docker on macOS can't use the Apple GPU, so this Ollama runs on CPU. For full speed install "
                    "the Ollama app (https://ollama.com/download), start it, and run `rack up` again.")
        gpu_on = gpu == "yes" or (gpu == "auto" and platform.system() != "Darwin" and bool(nvidia_gpus()))
        if gpu_on:
            files += ["-f", "docker-compose.gpu.yml"]
    if gateway:
        key = env.get("RACK_API_KEY", "")
        if len(key) < 16:
            key = secrets.token_urlsafe(32)
            set_env_value(root, "RACK_API_KEY", key)
            ui.ok("Generated an API key for the gateway (RACK_API_KEY in .env)")
        run_env["RACK_API_KEY"] = key
    profiles = ["--profile", "gateway"] if gateway else []
    cmd = base + files + profiles + ["up", "-d"] + services
    ui.step("Starting containers (the first run downloads a few GB of images)")
    r = subprocess.run(cmd, cwd=root, env=run_env)
    if r.returncode != 0 and gpu_on:
        ui.warn("Starting with GPU failed (is the NVIDIA Container Toolkit installed? "
                "https://docs.ollama.com/docker). Retrying on CPU...")
        cmd = base + ["-f", "docker-compose.yml"] + profiles + ["up", "-d"] + services
        gpu_on = False
        r = subprocess.run(cmd, cwd=root, env=run_env)
    if r.returncode != 0:
        raise RackError("docker compose up failed (see the output above).")

    if not use_native:
        bind = read_env(root).get("BIND_ADDR", "127.0.0.1")
        probe = Ollama(f"http://{'localhost' if bind in ('', '0.0.0.0', '127.0.0.1') else bind}:11434", timeout=5)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                probe.version()
                break
            except OllamaError:
                time.sleep(2)
        else:
            raise RackError("Ollama didn't come up within 2 minutes. Check `docker compose logs ollama`.")
    env = read_env(root)
    return {"native": use_native, "gpu": gpu_on, "gateway": gateway,
            "webui_port": env.get("OPEN_WEBUI_PORT", "3000"), "gateway_port": env.get("GATEWAY_PORT", "8000"),
            "bind": env.get("BIND_ADDR", "127.0.0.1")}


def down(root: Path) -> None:
    cmd = compose_cmd() + ["-f", "docker-compose.yml", "--profile", "gateway", "down"]
    env = os.environ.copy()
    env.setdefault("RACK_API_KEY", read_env(root).get("RACK_API_KEY", ""))
    r = subprocess.run(cmd, cwd=root, env=env)
    if r.returncode != 0:
        raise RackError("docker compose down failed (see the output above).")
