"""Platform settings (rack.toml at the repo root) with env-var overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore

DEFAULTS = {
    "server": {
        "ollama_url": "http://localhost:11434",
        "api_key": "",
        "webui_url": "http://localhost:3000",
    },
    "kaggle": {
        "username": "",
        "accelerator": "NvidiaTeslaT4",
        "timeout_minutes": 240,
        "poll_seconds": 60,
    },
    "local_train": {
        "python": "python",
    },
}

ENV_OVERRIDES = {
    ("server", "ollama_url"): "RACK_OLLAMA_URL",
    ("server", "api_key"): "RACK_API_KEY",
    ("server", "webui_url"): "RACK_WEBUI_URL",
    ("kaggle", "username"): "KAGGLE_USERNAME",
    ("kaggle", "timeout_minutes"): "RACK_KAGGLE_TIMEOUT_MINUTES",
    ("local_train", "python"): "RACK_TRAIN_PYTHON",
}


class RackError(Exception):
    """An error with a message meant for the user (no traceback)."""


def find_root(start: Path | None = None) -> Path:
    """The platform root is the nearest folder (walking up) that holds rack.toml."""
    env = os.environ.get("RACK_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "rack.toml").is_file():
            return p
    raise RackError(
        "Couldn't find rack.toml. Run rack from inside your platform folder "
        "(the one with rack.toml), or set RACK_HOME to it."
    )


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise RackError(f"{path} isn't valid TOML: {e}") from e


def load_platform(root: Path) -> dict:
    cfg = deep_merge(DEFAULTS, load_toml(root / "rack.toml"))
    for (section, key), env in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg[section][key] = os.environ[env]
    cfg["server"]["ollama_url"] = cfg["server"]["ollama_url"].rstrip("/")
    cfg["_root"] = str(root)
    return cfg
