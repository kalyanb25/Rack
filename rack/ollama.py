"""Minimal Ollama REST client (standard library only).

Talks to the documented HTTP API, so it works the same against a native Ollama
install, the Docker container, or a remote box over Tailscale.
Docs: https://docs.ollama.com/api/introduction
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from .config import RackError


class OllamaError(RackError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _error_text(e: urllib.error.HTTPError) -> str:
    try:
        raw = e.read().decode("utf-8", "replace")
    except Exception:
        raw = ""
    try:
        return json.loads(raw).get("error") or raw or e.reason
    except Exception:
        return raw.strip() or str(e.reason)


def file_digest(path: Path, progress: Callable[[int, int], None] | None = None) -> str:
    """sha256 of a file as Ollama expects it: 'sha256:<hex>'."""
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    return "sha256:" + h.hexdigest()


class _ProgressReader:
    """File wrapper that reports bytes read, so uploads can show progress."""

    def __init__(self, f, total: int, progress: Callable[[int, int], None] | None):
        self._f, self._total, self._progress, self._done = f, total, progress, 0

    def read(self, n: int = -1) -> bytes:
        chunk = self._f.read(n)
        self._done += len(chunk)
        if self._progress:
            self._progress(self._done, self._total)
        return chunk


class Ollama:
    def __init__(self, url: str, api_key: str = "", timeout: float = 600):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # Ollama is self-hosted, so skip any corporate HTTP proxy unless asked not to.
        handlers = [] if os.environ.get("RACK_USE_PROXY") else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    # ---- plumbing -------------------------------------------------------
    def _open(self, method: str, path: str, body: dict | None = None, *, data=None,
              headers: dict | None = None, timeout: float | None = None):
        hdrs = {"Accept": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=hdrs)
        try:
            return self._opener.open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raise OllamaError(e.code, _error_text(e)) from None
        except urllib.error.URLError as e:
            raise OllamaError(0, f"Can't reach Ollama at {self.url} ({e.reason}). "
                                 "Is it running? Try `rack up`, or check server.ollama_url in rack.toml.") from None
        except (ConnectionError, socket.timeout, TimeoutError) as e:
            raise OllamaError(0, f"Connection to Ollama at {self.url} failed: {e}") from None

    def _json(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        with self._open(method, path, body, timeout=timeout) as r:
            raw = r.read()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw.decode("utf-8", "replace")}

    def _stream(self, method: str, path: str, body: dict, timeout: float | None = None) -> Iterator[dict]:
        r = self._open(method, path, body, timeout=timeout)
        with r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise OllamaError(500, obj["error"])
                yield obj

    # ---- endpoints ------------------------------------------------------
    def version(self) -> str:
        return self._json("GET", "/api/version", timeout=10).get("version", "?")

    def tags(self) -> list[dict]:
        return self._json("GET", "/api/tags", timeout=30).get("models", [])

    def ps(self) -> list[dict]:
        return self._json("GET", "/api/ps", timeout=30).get("models", [])

    def show(self, model: str) -> dict:
        return self._json("POST", "/api/show", {"model": model}, timeout=60)

    def exists(self, model: str) -> bool:
        try:
            self.show(model)
            return True
        except OllamaError as e:
            if e.status == 404:
                return False
            raise

    def pull(self, model: str) -> Iterator[dict]:
        return self._stream("POST", "/api/pull", {"model": model, "stream": True})

    def create(self, model: str, **fields) -> Iterator[dict]:
        body: dict = {"model": model, "stream": True}
        body.update({k: v for k, v in fields.items() if v not in (None, "", [], {})})
        return self._stream("POST", "/api/create", body)

    def copy(self, source: str, destination: str) -> None:
        self._json("POST", "/api/copy", {"source": source, "destination": destination})

    def delete(self, model: str) -> None:
        self._json("DELETE", "/api/delete", {"model": model})

    def chat(self, model: str, messages: list[dict], *, options: dict | None = None,
             stream: bool = False, keep_alive=None, think=None):
        body: dict = {"model": model, "messages": messages, "stream": stream}
        if options:
            body["options"] = options
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        if think is not None:
            body["think"] = think
        if stream:
            return self._stream("POST", "/api/chat", body)
        return self._json("POST", "/api/chat", body)

    def blob_exists(self, digest: str) -> bool:
        try:
            with self._open("HEAD", f"/api/blobs/{digest}", timeout=60):
                return True
        except OllamaError as e:
            if e.status == 404:
                return False
            raise

    def push_blob(self, path: Path, digest: str,
                  progress: Callable[[int, int], None] | None = None) -> None:
        size = path.stat().st_size
        with open(path, "rb") as f:
            reader = _ProgressReader(f, size, progress)
            with self._open("POST", f"/api/blobs/{digest}", data=reader,
                            headers={"Content-Type": "application/octet-stream",
                                     "Content-Length": str(size)},
                            timeout=3600):
                pass
