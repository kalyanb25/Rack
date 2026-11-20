"""The core operations: build, import, deploy, rollback, remove, eval.

The CLI is a thin layer over these, and the training runner reuses import_gguf.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import modelfile as mf
from . import ui
from .config import RackError
from .evals import load_cases, run_eval
from .ollama import Ollama, OllamaError, file_digest
from .project import Project
from .registry import Registry, now_iso


def _consume(events, label: str) -> None:
    """Show progress for streaming pull/create events."""
    bar = ui.Progress(label)
    last_status = ""
    for ev in events:
        status = ev.get("status", "")
        if ev.get("total"):
            bar.update(ev.get("completed", 0), ev["total"], label=f"{label}: {status[:40]}")
        elif status and status != last_status:
            bar.update(0, None, label=f"{label}: {status[:60]}")
        last_status = status or last_status
    bar.finish()


def ensure_base(client: Ollama, base: str) -> None:
    if client.exists(base):
        return
    ui.step(f"Downloading base model {base} (one time)...")
    _consume(client.pull(base), f"pull {base}")
    ui.ok(f"{base} ready")


def build_prompt_version(client: Ollama, project: Project, notes: str = "", force: bool = False) -> dict:
    """New version = base model + system prompt + parameters + few-shot examples. No training."""
    reg = Registry(project.dir, project.name)
    fp = project.prompt_fingerprint()
    for v in reversed(reg.live()):
        if v.get("kind") == "prompt" and v.get("fingerprint") == fp and not force:
            raise RackError(f"Nothing changed since {project.name}:{v['version']} "
                            f"(same base, system prompt, parameters and examples). "
                            f"Edit models/{project.name}/system.md or model.toml, or pass --force.")
    ensure_base(client, project.base_ollama)
    version = reg.next_version()
    tag = project.tag(version)
    system = project.system_prompt()
    ui.step(f"Building {tag} from {project.base_ollama}")
    _consume(client.create(tag, **{"from": project.base_ollama}, system=system or None,
                           parameters=project.parameters() or None,
                           messages=project.examples() or None), f"create {tag}")
    entry = reg.add({"version": version, "tag": tag, "kind": "prompt", "base": project.base_ollama,
                     "fingerprint": fp, "notes": notes})
    reg.save()
    ui.ok(f"Built {tag}")
    return entry


def import_gguf(client: Ollama, project: Project, gguf: Path, *, modelfile: Path | None = None,
                kind: str = "import", notes: str = "", train: dict | None = None,
                base: str | None = None) -> dict:
    """New version from a GGUF file (a fine-tune, or any GGUF you downloaded)."""
    gguf = Path(gguf)
    if not gguf.is_file():
        raise RackError(f"GGUF file not found: {gguf}")
    parsed = mf.parse(modelfile.read_text(encoding="utf-8")) if modelfile and modelfile.is_file() else None

    size = gguf.stat().st_size
    bar = ui.Progress(f"hashing {gguf.name}", size)
    digest = file_digest(gguf, lambda d, t: bar.update(d, t))
    bar.finish()
    if client.blob_exists(digest):
        ui.ok("Server already has these weights, skipping upload")
    else:
        bar = ui.Progress(f"uploading {gguf.name}", size)
        client.push_blob(gguf, digest, lambda d, t: bar.update(d, t))
        bar.finish(f"Uploaded {ui.human_bytes(size)}")

    reg = Registry(project.dir, project.name)
    version = reg.next_version()
    tag = project.tag(version)

    params = dict(parsed["parameters"]) if parsed else {}
    params.update(project.parameters())  # your model.toml settings win
    system = project.system_prompt() or (parsed or {}).get("system")
    template = (parsed or {}).get("template")
    if "stop" not in params and mf.infer_stops(template):
        params["stop"] = mf.infer_stops(template)  # safety net so replies end where they should
    _consume(client.create(tag, files={gguf.name: digest}, template=template or None,
                           system=system or None, parameters=params or None,
                           messages=project.examples() or None), f"create {tag}")

    template_source = "?"
    try:
        template_source = client.show(tag).get("template_source", "?")
    except OllamaError:
        pass
    if template_source == "none":
        ui.warn("Ollama found no chat template for this GGUF. Chat may not work properly. "
                "Import it with the Modelfile produced by training (--modelfile).")

    entry = reg.add({"version": version, "tag": tag, "kind": kind, "base": base or project.base_hf,
                     "gguf": {"file": gguf.name, "sha256": digest.split(":", 1)[1][:16], "bytes": size},
                     "template_source": template_source, "notes": notes})
    if train:
        entry["train"] = train
    reg.save()
    ui.ok(f"Created {tag} from {gguf.name}")
    return entry


def deploy(client: Ollama, project: Project, spec: str | None, action: str = "deploy") -> dict:
    reg = Registry(project.dir, project.name)
    v = reg.get(spec)
    if not client.exists(v["tag"]):
        raise RackError(f"{v['tag']} is in the registry but not on the Ollama server at {client.url}. "
                        "If you switched servers, rebuild it (rack build) or re-import it.")
    stable = project.tag("latest")
    client.copy(v["tag"], stable)
    reg.record_deploy(v["version"], action)
    reg.save()
    ui.ok(f"{project.name} {ui.arrow()} {v['version']}  (apps and the web UI use \"{project.name}\")")
    return v


def rollback(client: Ollama, project: Project) -> dict:
    reg = Registry(project.dir, project.name)
    prev = reg.previous_deploy()
    if not prev:
        raise RackError(f"No earlier deploy of {project.name} to roll back to. "
                        f"Use `rack deploy {project.name} <version>`.")
    return deploy(client, project, prev, action="rollback")


def remove_version(client: Ollama, project: Project, spec: str) -> None:
    reg = Registry(project.dir, project.name)
    v = reg.get(spec)
    if v["version"] == reg.deployed:
        raise RackError(f"{v['tag']} is deployed. Deploy another version first.")
    try:
        client.delete(v["tag"])
    except OllamaError as e:
        if e.status != 404:
            raise
    v["removed"] = True
    v["removed_at"] = now_iso()
    reg.save()
    ui.ok(f"Removed {v['tag']}")


def evaluate(client: Ollama, project: Project, spec: str | None, *, verbose: bool = False,
             quiet: bool = False) -> dict:
    reg = Registry(project.dir, project.name)
    v = reg.get(spec)
    ecfg = project.cfg["eval"]
    cases = load_cases(project.path(ecfg["cases"]))
    options = {"temperature": float(ecfg.get("temperature", 0.0)), "seed": int(ecfg.get("seed", 42))}
    think = {"on": True, "off": False}.get(str(ecfg.get("think", "default")).lower())
    judge = ecfg.get("judge") or ""
    if judge:
        ensure_base(client, judge)
    ui.step(f"Evaluating {v['tag']} on {len(cases)} case(s)" + (f", judge: {judge}" if judge else ""))

    def on_case(row):
        if quiet:
            return
        mark = ui.green("PASS") if row["pass"] else ui.red("FAIL")
        why = "" if row["pass"] else "  " + ui.dim("; ".join(c["check"] + (f" ({c['detail']})" if c["detail"] else "")
                                                       for c in row["checks"] if not c["pass"]))
        print(f"  {mark}  {row['id']}{why}")
        if verbose:
            print(ui.dim("        " + row["answer"][:300].replace("\n", " ")))

    result = run_eval(client, v["tag"], cases, options=options, judge_model=judge, think=think, on_case=on_case)
    result["version"] = v["version"]
    result["at"] = now_iso()
    out_dir = project.dir / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{v['version']}-{stamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    reg.record_eval(v["version"], {"score": result["score"], "passed": result["passed"],
                                   "total": result["total"], "avg_latency_ms": result["avg_latency_ms"],
                                   "at": result["at"]})
    reg.save()
    score = result["score"]
    color = ui.green if score >= 0.8 else ui.yellow if score >= 0.5 else ui.red
    ui.info(f"  {v['tag']}: {color(ui.pct(score))} ({result['passed']}/{result['total']} passed, "
            f"avg {result['avg_latency_ms']} ms)")
    return result
