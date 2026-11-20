"""Model projects: one folder per custom model under models/<name>/."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import data
from .config import RackError, deep_merge, load_toml

DEFAULT_OLLAMA_BASE = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_HF_BASE = "unsloth/Qwen3-4B-Instruct-2507"

PROJECT_DEFAULTS = {
    "name": "",
    "description": "",
    "base": {"ollama": DEFAULT_OLLAMA_BASE, "hf": DEFAULT_HF_BASE},
    "prompt": {"system": "system.md", "examples": "examples.jsonl"},
    "parameters": {},
    "train": {
        "data": "data/train.jsonl",
        "valid": "",
        "backend": "auto",
        "epochs": 2,
        "max_steps": 0,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 16,
        "max_seq_length": 2048,
        "batch_size": 2,
        "grad_accum": 4,
        "quantization": "q4_k_m",
        "include_system": True,
        "chat_template": "auto",
        "load_in_4bit": True,
    },
    "eval": {
        "cases": "evals/cases.jsonl",
        "judge": "",
        "temperature": 0.0,
        "seed": 42,
        "think": "default",
    },
}

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")


def validate_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise RackError(f"Bad model name {name!r}: use 1-32 lowercase letters, digits, '.', '_' or '-', "
                        "starting with a letter or digit (e.g. my-assistant).")
    return name


def models_dir(root: Path) -> Path:
    return root / "models"


def list_projects(root: Path) -> list[str]:
    d = models_dir(root)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "model.toml").is_file())


class Project:
    def __init__(self, root: Path, name: str):
        self.root = root
        self.name = validate_name(name)
        self.dir = models_dir(root) / name
        if not (self.dir / "model.toml").is_file():
            known = ", ".join(list_projects(root)) or "none yet"
            raise RackError(f"No model project called {name!r} (models/{name}/model.toml). "
                            f"Existing: {known}. Create one with `rack new {name}`.")
        cfg = deep_merge(PROJECT_DEFAULTS, load_toml(self.dir / "model.toml"))
        cfg["name"] = name
        self.cfg = cfg

    # ---- paths ----------------------------------------------------------
    def path(self, rel: str) -> Path:
        return (self.dir / rel) if rel else self.dir / "__missing__"

    @property
    def base_ollama(self) -> str:
        return self.cfg["base"]["ollama"]

    @property
    def base_hf(self) -> str:
        return self.cfg["base"]["hf"]

    def tag(self, version: str) -> str:
        return f"{self.name}:{version}"

    # ---- content --------------------------------------------------------
    def system_prompt(self) -> str:
        p = self.path(self.cfg["prompt"].get("system", ""))
        return p.read_text(encoding="utf-8").strip() if p.is_file() else ""

    def examples(self) -> list[dict]:
        """Few-shot example turns baked into the model (Ollama MESSAGE entries)."""
        p = self.path(self.cfg["prompt"].get("examples", ""))
        if not p.is_file():
            return []
        exs, errors = data.load(p)
        if errors:
            line, why = errors[0]
            raise RackError(f"{p}: line {line}: {why}")
        out: list[dict] = []
        for msgs in exs:
            out.extend(m for m in msgs if m["role"] != "system")
        return out

    def parameters(self) -> dict:
        params = dict(self.cfg.get("parameters", {}))
        if "stop" in params and isinstance(params["stop"], str):
            params["stop"] = [params["stop"]]
        return params

    def prompt_fingerprint(self) -> str:
        blob = json.dumps({"base": self.base_ollama, "system": self.system_prompt(),
                           "parameters": self.parameters(), "examples": self.examples()},
                          sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


MODEL_TOML = '''# {name}: one of your custom models.
name = "{name}"
description = "{description}"

[base]
# The open-weight model you start from.
#   ollama: used by `rack build` (system prompt + settings on top of the base, no training)
#   hf:     used by `rack train` (LoRA fine-tune on your data). Keep both the same model family.
ollama = "{base_ollama}"
hf = "{base_hf}"

[prompt]
system = "system.md"          # the model's built-in instructions
examples = "examples.jsonl"   # optional few-shot conversations baked into the model

[parameters]
# Any Ollama model parameter: temperature, top_p, top_k, num_ctx, repeat_penalty, stop, seed...
temperature = 0.7
num_ctx = 8192

[train]
data = "data/train.jsonl"     # chat-format JSONL, see `rack data check {name}`
valid = ""                    # optional held-out set
backend = "auto"              # auto | kaggle (free cloud GPU) | local (your NVIDIA GPU)
epochs = 2
learning_rate = 2e-4
lora_r = 16
max_seq_length = 2048
quantization = "q4_k_m"       # GGUF quant for serving: q4_k_m (small), q5_k_m, q8_0 (best quality)
include_system = true         # add the system prompt to training examples that don't have one

[eval]
cases = "evals/cases.jsonl"   # test prompts + checks, see `rack eval {name}`
judge = ""                    # optional: another model that grades rubric checks, e.g. "qwen3:8b"
temperature = 0.0             # deterministic answers make versions comparable
'''

SYSTEM_MD = """You are {name}, a helpful assistant.

Be concise and direct. If you don't know something, say so instead of guessing.
"""

TRAIN_JSONL = [
    {"messages": [
        {"role": "user", "content": "Replace this with a real question your model should handle."},
        {"role": "assistant", "content": "Replace this with the exact answer you want, in the voice you want."}]},
    {"messages": [
        {"role": "user", "content": "Each line is one conversation. Add 100+ of these."},
        {"role": "assistant", "content": "Quality beats quantity: every answer here is something the model will imitate."}]},
]

CASES_JSONL = [
    {"id": "says-name", "prompt": "Who are you?", "contains": ["{name}"]},
    {"id": "stays-short", "prompt": "Explain DNS in one sentence.", "max_words": 40},
    {"id": "admits-unknown", "prompt": "What did I eat for breakfast today?",
     "contains_any": ["don't know", "do not know", "can't know", "cannot know", "no way to know", "not sure"]},
]


def new_project(root: Path, name: str, base_ollama: str = DEFAULT_OLLAMA_BASE,
                base_hf: str = DEFAULT_HF_BASE, description: str = "") -> Path:
    validate_name(name)
    d = models_dir(root) / name
    if (d / "model.toml").exists():
        raise RackError(f"models/{name} already exists.")
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "evals").mkdir(parents=True, exist_ok=True)
    desc = description.replace('"', "'") or f"My custom model {name}"
    (d / "model.toml").write_text(MODEL_TOML.format(name=name, description=desc, base_ollama=base_ollama,
                                                    base_hf=base_hf), encoding="utf-8")
    (d / "system.md").write_text(SYSTEM_MD.format(name=name), encoding="utf-8")
    (d / "examples.jsonl").write_text("", encoding="utf-8")
    (d / "data" / "train.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in TRAIN_JSONL), encoding="utf-8")
    cases = [json.loads(json.dumps(c).replace("{name}", name)) for c in CASES_JSONL]
    (d / "evals" / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    return d
