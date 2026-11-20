"""In-process fake of the Ollama REST API, written from https://docs.ollama.com/api.

Enough behaviour to exercise rack end to end: pull, create (from a model or from uploaded
GGUF blobs), blobs, show, copy, delete, tags, ps, chat (streaming and not).
Chat answers are deterministic: "[<first line of system prompt>] <last user message>".
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LIBRARY = {"qwen3:4b-instruct-2507-q4_K_M", "qwen3:8b", "qwen3:1.7b", "judge:latest"}


def norm(name: str) -> str:
    return name if ":" in name else name + ":latest"


class State:
    def __init__(self):
        self.models: dict[str, dict] = {}
        self.blobs: dict[str, int] = {}
        self.log: list[tuple[str, str, dict]] = []
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    state: State  # set on the subclass

    def log_message(self, *args):  # keep test output quiet
        pass

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw) if raw else {}

    def _send(self, code: int, obj: dict | None = None):
        data = json.dumps(obj).encode() if obj is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data and self.command != "HEAD":
            self.wfile.write(data)

    def _stream(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for ev in events:
            line = (json.dumps(ev) + "\n").encode()
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    # -- routes --
    def do_GET(self):
        s = self.state
        if self.path == "/api/version":
            return self._send(200, {"version": "0.40.16-fake"})
        if self.path == "/api/tags":
            with s.lock:
                ms = [{"name": n, "model": n, "size": m.get("size", 1000),
                       "modified_at": "2026-11-20T12:00:00Z", "digest": "abc"} for n, m in s.models.items()]
            return self._send(200, {"models": ms})
        if self.path == "/api/ps":
            return self._send(200, {"models": []})
        self._send(404, {"error": "not found"})

    def do_HEAD(self):
        if self.path.startswith("/api/blobs/"):
            digest = self.path.rsplit("/", 1)[1]
            return self._send(200 if digest in self.state.blobs else 404)
        self._send(404)

    def do_DELETE(self):
        b = self._body()
        if self.path == "/api/delete":
            name = norm(b.get("model", ""))
            with self.state.lock:
                if name not in self.state.models:
                    return self._send(404, {"error": f"model '{name}' not found"})
                del self.state.models[name]
            return self._send(200)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        s = self.state
        if self.path.startswith("/api/blobs/"):
            digest = self.path.rsplit("/", 1)[1]
            n = int(self.headers.get("Content-Length") or 0)
            h, left = hashlib.sha256(), n
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                h.update(chunk)
                left -= len(chunk)
            if "sha256:" + h.hexdigest() != digest:
                return self._send(400, {"error": "digest mismatch"})
            s.blobs[digest] = n
            return self._send(201)

        b = self._body()
        s.log.append(("POST", self.path, b))
        if self.path == "/api/show":
            name = norm(b.get("model", ""))
            m = s.models.get(name)
            if not m:
                return self._send(404, {"error": f"model '{name}' not found"})
            src = "go_template" if m.get("template") else ("gguf_chat_template" if m.get("files") else "go_template")
            return self._send(200, {"parameters": json.dumps(m.get("parameters", {})), "template": m.get("template", ""),
                                    "template_source": src, "capabilities": ["completion"],
                                    "details": {"format": "gguf", "family": "fake"}})
        if self.path == "/api/pull":
            name = norm(b["model"])
            if name not in LIBRARY:
                return self._stream([{"status": "pulling manifest"},
                                     {"error": "pull model manifest: file does not exist"}])
            with s.lock:
                s.models[name] = {"system": "", "size": 2_500_000_000}
            return self._stream([{"status": "pulling manifest"},
                                 {"status": "pulling abc", "digest": "sha256:abc", "total": 100, "completed": 50},
                                 {"status": "pulling abc", "digest": "sha256:abc", "total": 100, "completed": 100},
                                 {"status": "verifying sha256 digest"}, {"status": "success"}])
        if self.path == "/api/create":
            name = norm(b["model"])
            if b.get("from"):
                base = s.models.get(norm(b["from"]))
                if not base:
                    return self._stream([{"error": f"base model '{b['from']}' not found"}])
                m = dict(base)
            elif b.get("files"):
                missing = [d for d in b["files"].values() if d not in s.blobs]
                if missing:
                    return self._stream([{"error": f"blob {missing[0]} not found"}])
                m = {"files": b["files"], "size": sum(s.blobs[d] for d in b["files"].values())}
            else:
                return self._stream([{"error": "neither 'from' or 'files' was specified"}])
            for k in ("system", "template", "parameters", "messages"):
                if k in b:
                    m[k] = b[k]
            with s.lock:
                s.models[name] = m
            return self._stream([{"status": "using existing layer sha256:abc"},
                                 {"status": "writing manifest"}, {"status": "success"}])
        if self.path == "/api/copy":
            src, dst = norm(b.get("source", "")), norm(b.get("destination", ""))
            with s.lock:
                if src not in s.models:
                    return self._send(404, {"error": f"model '{src}' not found"})
                s.models[dst] = dict(s.models[src])
            return self._send(200, {"status": "success"})
        if self.path == "/api/chat":
            name = norm(b.get("model", ""))
            m = s.models.get(name)
            if not m:
                return self._send(404, {"error": f"model '{name}' not found"})
            user = [x["content"] for x in b["messages"] if x["role"] == "user"][-1]
            if name == "judge:latest":
                answer = "FAIL the response is bad" if "FAIL_ME" in user else "PASS looks right"
            else:
                sys_line = (m.get("system") or "no system").splitlines()[0]
                answer = f"[{sys_line}] {user}"
            if b.get("stream", True):
                words = answer.split(" ")
                evs = [{"model": name, "message": {"role": "assistant", "content": w + (" " if i < len(words) - 1 else "")},
                        "done": False} for i, w in enumerate(words)]
                evs.append({"model": name, "message": {"role": "assistant", "content": ""}, "done": True})
                return self._stream(evs)
            time.sleep(0.001)
            return self._send(200, {"model": name, "message": {"role": "assistant", "content": answer},
                                    "done": True, "eval_count": len(answer.split())})
        self._send(404, {"error": "not found"})


class FakeOllama:
    def __init__(self):
        self.state = State()
        handler = type("H", (Handler,), {"state": self.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
