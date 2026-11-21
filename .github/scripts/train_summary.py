"""Markdown summary of a training run for the GitHub Actions job page.

Usage: python .github/scripts/train_summary.py <model-name>  >> $GITHUB_STEP_SUMMARY
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def loss_trend(metrics: dict) -> str:
    losses = [h["loss"] for h in metrics.get("loss_history") or [] if isinstance(h.get("loss"), (int, float))]
    if len(losses) < 2:
        return "n/a"
    k = max(1, len(losses) // 10)
    first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
    return f"{first:.3f} → {last:.3f} over {len(losses)} steps"


FEW_STEPS = 20


def cell(text, n: int = 110) -> str:
    """One markdown table cell: single line, no pipes, trimmed."""
    t = " ".join(str(text or "").split()).replace("|", "/")
    return t if len(t) <= n else t[: n - 1] + "…"


def failed_cases(models_dir: Path) -> list[str]:
    """Which eval cases each version failed, and why, from the latest result file per version."""
    latest: dict[str, Path] = {}
    for f in sorted((models_dir / "evals" / "results").glob("v*-*.json")):
        latest[f.name.split("-", 1)[0]] = f  # names sort by timestamp, so the last one wins
    rows = []
    for version in sorted(latest, key=lambda v: int(v[1:]) if v[1:].isdigit() else 0):
        res = load(latest[version]) or {}
        for c in res.get("cases") or []:
            if c.get("pass"):
                continue
            why = "; ".join(ch.get("check", "?") + (f" ({ch['detail']})" if ch.get("detail") else "")
                            for ch in c.get("checks") or [] if not ch.get("pass"))
            rows.append(f"| {cell(c.get('id'), 40)} | {version} | {cell(why, 70)} | {cell(c.get('answer'))} |")
    if not rows:
        return []
    return ["", "<details><summary>Failed eval cases</summary>", "",
            "| Case | Version | Failed check | Answer starts |", "|---|---|---|---|", *rows, "", "</details>"]


def main(name: str, root: Path = Path(".")) -> str:
    lines = [f"## Fine-tune: `{name}`", ""]
    metrics = load(root / "dist" / "metrics.json")
    if metrics is None:
        jobs = sorted((root / ".rack" / "jobs").glob(f"{name}-*/output/metrics.json"))
        metrics = load(jobs[-1]) if jobs else None
    if metrics is None:
        lines.append("Training didn't produce results. Check the **Fine-tune on Kaggle's free GPU** step log above.")
    elif metrics.get("status") != "ok":
        lines += [f"**Training failed:** `{metrics.get('error')}`", "", "```",
                  "\n".join((metrics.get("traceback") or "").strip().splitlines()[-15:]), "```"]
    else:
        loss = metrics.get("final_loss")
        steps = metrics.get("steps")
        lines += [
            f"- GPU: {metrics.get('gpu')}",
            f"- Examples: {metrics.get('examples')}",
            f"- Training time: {metrics.get('train_runtime_s')} s",
            f"- Training steps: {steps if steps is not None else 'n/a'}"
            + (" (packed rows)" if metrics.get("packing") else ""),
            f"- Final loss: {round(loss, 4) if isinstance(loss, float) else loss}",
            f"- Loss trend: {loss_trend(metrics)}",
            f"- Model file: `{metrics.get('gguf')}`",
        ]
        labels = metrics.get("labels") or {}
        if labels.get("tokens"):
            share = round(100 * labels["trained_tokens"] / labels["tokens"], 1)
            lines.append(f"- Loss covers: {labels['trained_tokens']} of {labels['tokens']} tokens ({share}%), "
                         f"first example trains on: “{cell(labels.get('trained_preview'), 90)}”")
        if isinstance(steps, int) and steps < FEW_STEPS:
            lines += ["", f"**Warning:** only {steps} training steps, too few to change the model much. "
                          "Add examples or epochs, or lower `grad_accum` in model.toml."]
        sample = metrics.get("sample")
        if sample:
            lines += ["", f"**Sample after training** (a question from the training set): {sample.get('prompt')}",
                      "", "> " + str(sample.get("answer", "")).replace("\n", "\n> ")]

    reg = load(root / "models" / name / "registry.json")
    if reg and reg.get("versions"):
        lines += ["", "| Version | Kind | Eval | Notes |", "|---|---|---|---|"]
        for v in reg["versions"]:
            ev = v.get("eval")
            score = f"{round(ev['score'] * 100)}% ({ev['passed']}/{ev['total']})" if ev else "-"
            lines.append(f"| {v['version']} | {v.get('kind')} | {score} | {v.get('notes', '')} |")
    lines += failed_cases(root / "models" / name)

    if metrics and metrics.get("status") == "ok":
        lines += ["", "### Use it on your machine",
                  f"1. Download the **{name}-finetune-run…** artifact at the bottom of this page and unzip it.",
                  f"2. With `rack up` running: `rack import {name} path/to/{metrics.get('gguf')}` "
                  "(the Modelfile next to it is picked up automatically).",
                  f"3. `rack eval {name}` and, if you like it, `rack deploy {name}`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(main(sys.argv[1]))
