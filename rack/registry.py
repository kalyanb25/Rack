"""Version registry for a model project (models/<name>/registry.json).

Plain JSON on purpose: easy to read, diff and commit to git.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import RackError


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Registry:
    def __init__(self, project_dir: Path, name: str):
        self.path = project_dir / "registry.json"
        self.name = name
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise RackError(f"{self.path} is corrupted: {e}") from e
        else:
            self.data = {"name": name, "versions": [], "deployed": None, "history": []}

    def save(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".registry-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)

    # ---- versions -------------------------------------------------------
    @property
    def versions(self) -> list[dict]:
        return self.data["versions"]

    def live(self) -> list[dict]:
        return [v for v in self.versions if not v.get("removed")]

    def next_version(self) -> str:
        nums = [int(m.group(1)) for v in self.versions if (m := re.fullmatch(r"v(\d+)", v["version"]))]
        return f"v{max(nums, default=0) + 1}"

    def add(self, entry: dict) -> dict:
        entry.setdefault("created_at", now_iso())
        self.versions.append(entry)
        return entry

    def get(self, spec: str | None) -> dict:
        """Resolve 'v3', '3', 'latest', 'deployed' (or None -> latest)."""
        live = self.live()
        if not live:
            raise RackError(f"{self.name} has no versions yet. Create one with "
                            f"`rack build {self.name}` or `rack train {self.name}`.")
        if spec in (None, "", "latest", "newest"):
            return live[-1]
        if spec in ("deployed", "prod", "current"):
            if not self.deployed:
                raise RackError(f"{self.name} isn't deployed yet. Run `rack deploy {self.name}`.")
            spec = self.deployed
        spec = spec if spec.startswith("v") else f"v{spec}"
        for v in live:
            if v["version"] == spec:
                return v
        raise RackError(f"{self.name} has no version {spec}. Try `rack versions {self.name}`.")

    # ---- deploys --------------------------------------------------------
    @property
    def deployed(self) -> str | None:
        return self.data.get("deployed")

    def record_deploy(self, version: str, action: str = "deploy") -> None:
        self.data["deployed"] = version
        self.data["history"].append({"at": now_iso(), "action": action, "version": version})

    def previous_deploy(self) -> str | None:
        """The version that was live before the current one (for rollback)."""
        current = self.deployed
        live_ids = {v["version"] for v in self.live()}
        for h in reversed(self.data["history"]):
            if h["version"] != current and h["version"] in live_ids:
                return h["version"]
        return None

    def record_eval(self, version: str, summary: dict) -> None:
        for v in self.versions:
            if v["version"] == version:
                v["eval"] = summary
                return
