"""rack command line.  Run `rack --help` or `rack <command> --help`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, data, doctor, ops, stack, ui
from .config import RackError, find_root, load_platform
from .evals import load_cases
from .ollama import Ollama, OllamaError
from .project import DEFAULT_HF_BASE, DEFAULT_OLLAMA_BASE, Project, list_projects, new_project
from .registry import Registry


class Ctx:
    def __init__(self):
        self.root = find_root()
        self.cfg = load_platform(self.root)
        self._client = None

    @property
    def client(self) -> Ollama:
        if self._client is None:
            s = self.cfg["server"]
            self._client = Ollama(s["ollama_url"], s.get("api_key", ""))
        return self._client

    def project(self, name: str) -> Project:
        return Project(self.root, name)


# ---------------------------------------------------------------- commands

def cmd_doctor(ctx: Ctx, a) -> None:
    info = doctor.detect(ctx.root)
    rec = doctor.recommend(info)
    gpus = ", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in info["gpus"]) or (
        "Apple Silicon GPU" if info["apple_silicon"] else "none detected")
    d = info["docker"]
    rows = [
        ["Machine", f"{info['os']} {info['arch']}, {info['cpus']} CPUs, {info['ram_gb'] or '?'} GB RAM"
                    + (" (WSL)" if info["wsl"] else "")],
        ["GPU", gpus],
        ["Disk free", f"{info['disk_free_gb']} GB"],
        ["Python", info["python"] + ("" if doctor.python_ok() else ui.red("  (need 3.10+)"))],
        ["Docker", ui.green("running") if d["running"] else (ui.yellow("installed, not running") if d["installed"]
                                                          else ui.red("not installed"))],
        ["Kaggle CLI", ("installed" if info["kaggle"]["cli"] else "not installed") + ", " +
                       ("token found" if info["kaggle"]["credentials"] else "no token")],
    ]
    try:
        v = ctx.client.version()
        rows.append(["Ollama", ui.green(f"v{v}") + f" at {ctx.client.url}"])
    except OllamaError:
        rows.append(["Ollama", ui.yellow(f"not reachable at {ctx.client.url}") + " (rack up starts it)"])
    for k, v in rows:
        print(f"  {ui.bold(k.ljust(11))} {v}")
    mem, tag, note = rec["serve"]
    print()
    print(ui.bold("What this machine can run"))
    print(f"  ~{rec['budget_gb']} GB usable on {rec['where']}.")
    print(f"  Recommended: {ui.cyan(tag)}  {ui.dim('(' + note + ')')}")
    others = [t for _, t, _ in rec["fits"] if t != tag]
    if others:
        print(ui.dim("  Also fits: " + ", ".join(others)))
    print(f"  Training: {rec['train'][1]}")
    if rec["tips"]:
        print()
        for t in rec["tips"]:
            ui.warn(t)


def cmd_up(ctx: Ctx, a) -> None:
    gpu = "yes" if a.gpu else "no" if a.no_gpu else "auto"
    native = "yes" if a.native else "no" if a.docker_ollama else "auto"
    res = stack.up(ctx.root, gpu=gpu, gateway=a.gateway, native=native)
    host = "localhost" if res["bind"] in ("127.0.0.1", "0.0.0.0", "") else res["bind"]
    print()
    ui.ok("rack is up" + (" (GPU)" if res["gpu"] else ""))
    print(f"  Web UI      http://{host}:{res['webui_port']}   {ui.dim('(first account you create becomes admin)')}")
    print(f"  Ollama API  {ctx.cfg['server']['ollama_url']}   {ui.dim('OpenAI-compatible at /v1')}")
    if res["gateway"]:
        print(f"  Gateway     http://{host}:{res['gateway_port']}   "
              f"{ui.dim('send: Authorization: Bearer $RACK_API_KEY (see .env)')}")
    print(ui.dim("  Containers restart automatically after reboots. Stop with `rack down`."))


def cmd_down(ctx: Ctx, a) -> None:
    stack.down(ctx.root)
    ui.ok("Stopped. Your models and chats are kept in Docker volumes.")


def cmd_status(ctx: Ctx, a) -> None:
    c = ctx.client
    try:
        v = c.version()
        loaded = c.ps()
        names = ", ".join(m.get("name", "?") for m in loaded) or "none"
        print(f"  {ui.bold('Server'.ljust(8))} {c.url}  {ui.green('online')}  Ollama v{v}")
        print(f"  {ui.bold('Loaded'.ljust(8))} {names}")
    except OllamaError as e:
        print(f"  {ui.bold('Server'.ljust(8))} {c.url}  {ui.red('offline')}  {ui.dim(e.message)}")
    print(f"  {ui.bold('Web UI'.ljust(8))} {ctx.cfg['server']['webui_url']}")
    print()
    rows = []
    for name in list_projects(ctx.root):
        try:
            reg = Registry(ctx.project(name).dir, name)
        except RackError as e:
            rows.append([name, "?", "?", "?", str(e)[:40]])
            continue
        live = reg.live()
        latest = live[-1] if live else None
        ev = latest.get("eval") if latest else None
        rows.append([name, str(len(live)), latest["version"] if latest else "-", reg.deployed or "-",
                     f"{latest['version']} {ui.pct(ev['score'])}" if ev else "-"])
    if rows:
        print(ui.table(rows, ["MODEL", "VERSIONS", "LATEST", "DEPLOYED", "LAST EVAL"]))
    else:
        print("  No model projects yet. Create one: rack new my-assistant")


def cmd_pull(ctx: Ctx, a) -> None:
    ops._consume(ctx.client.pull(a.model), f"pull {a.model}")
    ui.ok(f"{a.model} ready")


def cmd_new(ctx: Ctx, a) -> None:
    d = new_project(ctx.root, a.name, a.base, a.hf, a.description or "")
    ui.ok(f"Created {d.relative_to(ctx.root)}")
    print(f"""
  Next:
    1. Edit models/{a.name}/system.md      (who the model is, how it answers)
    2. rack build {a.name}                 (makes {a.name}:v1, no training needed)
    3. rack eval {a.name}                  (scores it on models/{a.name}/evals/cases.jsonl)
    4. rack deploy {a.name}                (now "{a.name}" shows up in the web UI and API)
  Later: add real examples to models/{a.name}/data/train.jsonl and run rack train {a.name}""")


def cmd_build(ctx: Ctx, a) -> None:
    p = ctx.project(a.name)
    entry = ops.build_prompt_version(ctx.client, p, notes=a.notes or "", force=a.force)
    _after_new_version(ctx, p, entry, a)


def cmd_import(ctx: Ctx, a) -> None:
    p = ctx.project(a.name)
    mfile = Path(a.modelfile) if a.modelfile else None
    if mfile is None and (Path(a.gguf).parent / "Modelfile").is_file():
        mfile = Path(a.gguf).parent / "Modelfile"
        ui.info(ui.dim(f"  using {mfile}"))
    entry = ops.import_gguf(ctx.client, p, Path(a.gguf), modelfile=mfile, kind="import", notes=a.notes or "")
    _after_new_version(ctx, p, entry, a)


def _after_new_version(ctx: Ctx, p: Project, entry: dict, a) -> None:
    if not getattr(a, "no_eval", False) and p.path(p.cfg["eval"]["cases"]).is_file():
        ops.evaluate(ctx.client, p, entry["version"])
    reg = Registry(p.dir, p.name)
    if getattr(a, "deploy", False) or not reg.deployed:
        if not reg.deployed and not getattr(a, "deploy", False):
            ui.info(ui.dim("  First version, deploying it so it shows up in the web UI."))
        ops.deploy(ctx.client, p, entry["version"])
    else:
        print(ui.dim(f"  Deployed is still {reg.deployed}. Promote this one with: rack deploy {p.name} "
                     f"{entry['version']}"))


def cmd_train(ctx: Ctx, a) -> None:
    from .train import runner
    if a.resume:
        entry = runner.resume(ctx.client, ctx.root, Path(a.resume), ctx.cfg, notes=a.notes or "")
        _after_new_version(ctx, ctx.project(entry["tag"].split(":")[0]), entry, a)
        return
    if not a.name:
        raise RackError("Which model? rack train <name>")
    p = ctx.project(a.name)
    report = data.check(p.path(p.cfg["train"]["data"]), int(p.cfg["train"]["max_seq_length"]))
    if report["errors"]:
        line, why = report["errors"][0]
        raise RackError(f"Fix your training data first: line {line}: {why} "
                        f"(`rack data check {p.name}` lists all {len(report['errors'])} problem(s)).")
    for w in report["warnings"]:
        ui.warn(w)
    job_dir = runner.prepare_job(ctx.root, p)
    ui.ok(f"Packaged {report['count']} examples {ui.arrow()} {job_dir.relative_to(ctx.root)}")
    if a.dry_run:
        print(f"""
  Dry run: nothing uploaded. To run this job anywhere with an NVIDIA GPU (Colab, a rented box...):
    python train_unsloth.py --job data/job.json --out output     (inside {job_dir})
  then bring the results back with:
    rack train --resume {job_dir}""")
        return
    backend = runner.resolve_backend(a.backend or p.cfg["train"]["backend"], ctx.cfg)
    note = a.notes or f"rack train {p.name}"
    if backend == "local":
        runner.run_local(job_dir, ctx.cfg)
    else:
        runner.run_kaggle(job_dir, p, ctx.cfg, note)
    entry = runner.collect(ctx.client, job_dir, p, notes=a.notes or "")
    _after_new_version(ctx, p, entry, a)


def cmd_versions(ctx: Ctx, a) -> None:
    p = ctx.project(a.name)
    reg = Registry(p.dir, p.name)
    rows = []
    for v in reg.versions:
        if v.get("removed") and not a.all:
            continue
        ev = v.get("eval")
        mark = ui.green(ui.sym("●", "*")) if v["version"] == reg.deployed else " "
        score = f"{ui.pct(ev['score'])} ({ev['passed']}/{ev['total']})" if ev else "-"
        extra = ""
        if v.get("train"):
            t = v["train"]
            extra = f"{t.get('examples')} ex, loss {round(t['final_loss'], 3) if isinstance(t.get('final_loss'), float) else '?'}"
        notes = " ".join(x for x in [extra, v.get("notes", "")] if x)
        rows.append([f"{mark} {v['version']}" + (" (removed)" if v.get("removed") else ""), v.get("kind", "?"),
                     v.get("base", "?")[:34], v.get("created_at", "")[:16].replace("T", " "), score, notes[:40]])
    if not rows:
        print(f"  {p.name} has no versions yet. Try: rack build {p.name}")
        return
    print(ui.table(rows, ["  VERSION", "KIND", "BASE", "CREATED (UTC)", "EVAL", "NOTES"]))
    if reg.deployed:
        print(ui.dim(f"  {ui.sym('●', '*')} = deployed ({p.name} {ui.arrow()} {reg.deployed})"))


def cmd_eval(ctx: Ctx, a) -> None:
    p = ctx.project(a.name)
    specs = a.versions or [None]
    compare = len(specs) > 1
    results = [ops.evaluate(ctx.client, p, s, verbose=a.verbose and not compare, quiet=compare) for s in specs]
    if len(results) > 1:
        print()
        ids = [c["id"] for c in results[0]["cases"]]
        rows = []
        for i, cid in enumerate(ids):
            rows.append([cid] + [ui.green("pass") if r["cases"][i]["pass"] else ui.red("FAIL") for r in results])
        rows.append([ui.bold("score")] + [ui.bold(ui.pct(r["score"])) for r in results])
        print(ui.table(rows, ["CASE"] + [r["version"] for r in results]))


def cmd_deploy(ctx: Ctx, a) -> None:
    ops.deploy(ctx.client, ctx.project(a.name), a.version)


def cmd_rollback(ctx: Ctx, a) -> None:
    ops.rollback(ctx.client, ctx.project(a.name))


def cmd_rm(ctx: Ctx, a) -> None:
    ops.remove_version(ctx.client, ctx.project(a.name), a.version)


def cmd_chat(ctx: Ctx, a) -> None:
    p = ctx.project(a.name)
    reg = Registry(p.dir, p.name)
    tag = reg.get(a.version)["tag"] if a.version else (p.tag("latest") if reg.deployed else reg.get(None)["tag"])
    print(ui.dim(f"Chatting with {tag}.  /reset starts over, /exit quits."
                 + ("  Conversations are saved to data/inbox.jsonl." if a.save else "")))
    history: list[dict] = []
    saved = 0
    while True:
        try:
            msg = input(ui.cyan(f"you {ui.sym('›', '>')} "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.strip() in ("/exit", "/quit"):
            break
        if msg.strip() == "/reset":
            saved += _save_chat(p, history) if a.save else 0
            history = []
            print(ui.dim("(new conversation)"))
            continue
        if not msg.strip():
            continue
        history.append({"role": "user", "content": msg})
        sys.stdout.write(ui.bold(f"model {ui.sym('›', '>')} "))
        reply, thinking_shown = [], False
        try:
            for ev in ctx.client.chat(tag, history, stream=True):
                m = ev.get("message", {})
                if m.get("thinking") and a.show_thinking:
                    sys.stdout.write(ui.dim(m["thinking"]))
                    thinking_shown = True
                if m.get("content"):
                    if thinking_shown:
                        sys.stdout.write("\n")
                        thinking_shown = False
                    sys.stdout.write(m["content"])
                    reply.append(m["content"])
                sys.stdout.flush()
        except KeyboardInterrupt:
            sys.stdout.write(ui.dim(" [stopped]"))
        print()
        history.append({"role": "assistant", "content": "".join(reply)})
    if a.save:
        saved += _save_chat(p, history)
        if saved:
            ui.ok(f"Saved {saved} conversation(s) to models/{p.name}/data/inbox.jsonl. Review them, keep the "
                  "good ones, and move them into train.jsonl.")


def _save_chat(p: Project, history: list[dict]) -> int:
    if not any(m["role"] == "assistant" and m["content"] for m in history):
        return 0
    inbox = p.dir / "data" / "inbox.jsonl"
    inbox.parent.mkdir(parents=True, exist_ok=True)
    with open(inbox, "a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": history}, ensure_ascii=False) + "\n")
    return 1


def cmd_data(ctx: Ctx, a) -> None:
    p = ctx.project(a.name)
    t = p.cfg["train"]
    paths = [p.path(t["data"])] + ([p.path(t["valid"])] if t.get("valid") else [])
    if a.file:
        paths = [Path(a.file)]
    bad = False
    for path in paths:
        r = data.check(path, int(t["max_seq_length"]))
        print(ui.bold(str(path)))
        print(f"  {r['count']} examples, {r['with_system']} with a system message, "
              f"~{r['avg_tokens']} tokens avg, ~{r['max_tokens']} max")
        for line, why in r["errors"][:25]:
            ui.fail(f"line {line}: {why}")
        if len(r["errors"]) > 25:
            ui.fail(f"...and {len(r['errors']) - 25} more")
        for w in r["warnings"]:
            ui.warn(w)
        if not r["errors"] and r["count"]:
            ui.ok("looks good")
        bad = bad or bool(r["errors"]) or not r["count"]
    try:
        n = len(load_cases(p.path(p.cfg["eval"]["cases"])))
        ui.ok(f"{n} eval case(s) in {p.cfg['eval']['cases']}")
    except RackError as e:
        ui.warn(str(e))
    return 1 if bad else 0


def cmd_models(ctx: Ctx, a) -> None:
    rows = [[m.get("name", "?"), ui.human_bytes(m.get("size", 0)), m.get("modified_at", "")[:16].replace("T", " ")]
            for m in ctx.client.tags()]
    print(ui.table(sorted(rows), ["NAME", "SIZE", "MODIFIED"]) if rows else "  No models on the server yet.")


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rack", description="Your own AI model platform: build, train, evaluate, "
                                 "deploy and use custom models on hardware you control.")
    ap.add_argument("--version", action="version", version=f"rack {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        return sp

    add("doctor", cmd_doctor, "check this machine and get model recommendations")
    sp = add("up", cmd_up, "start Ollama + the web UI (Docker)")
    sp.add_argument("--gpu", action="store_true", help="force NVIDIA GPU mode")
    sp.add_argument("--no-gpu", action="store_true", help="force CPU mode")
    sp.add_argument("--gateway", action="store_true", help="also start the API-key gateway on port 8000")
    sp.add_argument("--native", action="store_true", help="use the Ollama app already running on this machine")
    sp.add_argument("--docker-ollama", action="store_true", help="always run Ollama in Docker")
    add("down", cmd_down, "stop the containers (data is kept)")
    add("status", cmd_status, "server health and all your models")
    sp = add("pull", cmd_pull, "download a model from the Ollama library")
    sp.add_argument("model")
    add("models", cmd_models, "list every model on the server")

    sp = add("new", cmd_new, "create a new model project")
    sp.add_argument("name")
    sp.add_argument("--base", default=DEFAULT_OLLAMA_BASE, help=f"Ollama base model (default {DEFAULT_OLLAMA_BASE})")
    sp.add_argument("--hf", default=DEFAULT_HF_BASE, help=f"Hugging Face base for training (default {DEFAULT_HF_BASE})")
    sp.add_argument("--description", default="")

    for name, fn, help_ in (("build", cmd_build, "new version from your system prompt + settings (no training)"),
                            ("import", cmd_import, "new version from a GGUF file")):
        sp = add(name, fn, help_)
        sp.add_argument("name")
        if name == "import":
            sp.add_argument("gguf", help="path to the .gguf file")
            sp.add_argument("--modelfile", help="Modelfile with the chat template (auto-detected next to the GGUF)")
        else:
            sp.add_argument("--force", action="store_true", help="build even if nothing changed")
        sp.add_argument("--notes", help="what changed in this version")
        sp.add_argument("--deploy", action="store_true", help="deploy it right away")
        sp.add_argument("--no-eval", action="store_true", help="skip the automatic eval")

    sp = add("train", cmd_train, "fine-tune on your data (free Kaggle GPU or your own NVIDIA GPU)")
    sp.add_argument("name", nargs="?")
    sp.add_argument("--backend", choices=["auto", "kaggle", "local"], help="where to train (default from model.toml)")
    sp.add_argument("--dry-run", action="store_true", help="package the job without running it")
    sp.add_argument("--resume", metavar="JOB_DIR", help="collect a job that was interrupted or ran elsewhere")
    sp.add_argument("--notes", help="what changed in this version")
    sp.add_argument("--deploy", action="store_true", help="deploy the new version right away")
    sp.add_argument("--no-eval", action="store_true", help="skip the automatic eval")

    sp = add("versions", cmd_versions, "list a model's versions")
    sp.add_argument("name")
    sp.add_argument("--all", action="store_true", help="include removed versions")
    sp = add("eval", cmd_eval, "score versions on your eval cases (several = side-by-side)")
    sp.add_argument("name")
    sp.add_argument("versions", nargs="*", help="e.g. v1 v2 (default: latest)")
    sp.add_argument("-v", "--verbose", action="store_true", help="print each answer")
    sp = add("deploy", cmd_deploy, "point <name> at a version (default: latest)")
    sp.add_argument("name")
    sp.add_argument("version", nargs="?")
    sp = add("rollback", cmd_rollback, "go back to the previously deployed version")
    sp.add_argument("name")
    sp = add("rm", cmd_rm, "delete a version from the server")
    sp.add_argument("name")
    sp.add_argument("version")
    sp = add("chat", cmd_chat, "chat in the terminal")
    sp.add_argument("name")
    sp.add_argument("version", nargs="?", help="default: the deployed version")
    sp.add_argument("--save", action="store_true", help="save conversations to data/inbox.jsonl for future training")
    sp.add_argument("--show-thinking", action="store_true", help="show reasoning from thinking models")
    sp = add("data", cmd_data, "validate training data and eval cases")
    sp.add_argument("action", choices=["check"])
    sp.add_argument("name")
    sp.add_argument("--file", help="check a specific JSONL file instead")
    return ap


def main(argv: list[str] | None = None) -> int:
    ui.safe_streams()
    ap = build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 0
    try:
        return a.fn(Ctx(), a) or 0
    except RackError as e:
        ui.fail(str(e))
        return 1
    except KeyboardInterrupt:
        print()
        return 130
