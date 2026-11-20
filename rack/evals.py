"""Evals: run test prompts against a model version and score them.

Each line of evals/cases.jsonl is one case:
    {"id": "greeting",                     # optional, defaults to case-N
     "prompt": "Hi!",                      # or "messages": [...] for multi-turn
     "contains": ["Kalyan"],               # every string must appear (case-insensitive)
     "contains_any": ["hello", "hey"],     # at least one must appear
     "not_contains": ["As an AI"],         # none may appear
     "regex": "\\d{3}-\\d{4}",             # must match somewhere
     "max_words": 60, "min_words": 3,
     "equals": "42",                       # whole answer, trimmed, case-insensitive
     "json": true,                         # answer must parse as JSON
     "judge": "Politely refuses and suggests a safer alternative."}   # graded by a judge model
A case passes when all its checks pass. Score = passed cases / total cases.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .config import RackError
from .ollama import Ollama, OllamaError

CHECK_KEYS = ("contains", "contains_any", "not_contains", "regex", "max_words",
              "min_words", "equals", "json", "judge")

JUDGE_SYSTEM = ("You are a strict, fair grader. You will see a rubric and a response. "
                "Decide if the response satisfies the rubric. Reply with PASS or FAIL as the first word, "
                "then one short sentence explaining why.")


def load_cases(path: Path) -> list[dict]:
    if not path.is_file():
        raise RackError(f"No eval cases at {path}. Add some test prompts there (one JSON object per line).")
    cases = []
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                raise RackError(f"{path}: line {no}: invalid JSON ({e.msg})") from e
            if "prompt" not in c and "messages" not in c:
                raise RackError(f"{path}: line {no}: needs a \"prompt\" or \"messages\"")
            if not any(k in c for k in CHECK_KEYS):
                raise RackError(f"{path}: line {no}: has no checks (use contains, regex, max_words, judge...)")
            c.setdefault("id", f"case-{len(cases) + 1}")
            cases.append(c)
    if not cases:
        raise RackError(f"{path} is empty.")
    return cases


def _as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def strip_thinking(text: str) -> str:
    """Some models inline their reasoning as <think>...</think>; grade only the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_checks(case: dict, answer: str, judge_fn=None) -> list[dict]:
    low = answer.lower()
    words = len(answer.split())
    results = []

    def add(name, passed, detail=""):
        results.append({"check": name, "pass": bool(passed), "detail": detail})

    if "contains" in case:
        missing = [s for s in _as_list(case["contains"]) if s.lower() not in low]
        add("contains", not missing, f"missing: {missing}" if missing else "")
    if "contains_any" in case:
        opts = _as_list(case["contains_any"])
        hit = any(s.lower() in low for s in opts)
        add("contains_any", hit, "" if hit else f"none of: {opts}")
    if "not_contains" in case:
        found = [s for s in _as_list(case["not_contains"]) if s.lower() in low]
        add("not_contains", not found, f"found: {found}" if found else "")
    if "regex" in case:
        try:
            m = re.search(case["regex"], answer, flags=re.IGNORECASE | re.DOTALL)
        except re.error as e:
            raise RackError(f"Case {case['id']}: bad regex {case['regex']!r}: {e}") from e
        add("regex", m is not None, "" if m else f"no match for /{case['regex']}/")
    if "max_words" in case:
        add("max_words", words <= int(case["max_words"]), f"{words} words")
    if "min_words" in case:
        add("min_words", words >= int(case["min_words"]), f"{words} words")
    if "equals" in case:
        add("equals", answer.strip().lower() == str(case["equals"]).strip().lower(), "")
    if case.get("json"):
        try:
            json.loads(answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
            add("json", True)
        except json.JSONDecodeError:
            add("json", False, "not valid JSON")
    if "judge" in case:
        if judge_fn is None:
            add("judge", False, "no judge model set ([eval] judge in model.toml)")
        else:
            passed, why = judge_fn(case["judge"], answer)
            add("judge", passed, why)
    return results


def make_judge(client: Ollama, judge_model: str):
    def judge(rubric: str, answer: str) -> tuple[bool, str]:
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Rubric: {rubric}\n\nResponse to grade:\n<<<\n{answer}\n>>>"}]
        r = client.chat(judge_model, msgs, options={"temperature": 0, "seed": 1})
        verdict = strip_thinking(r.get("message", {}).get("content", "")).strip()
        first = re.split(r"\W+", verdict, maxsplit=1)[0].upper() if verdict else ""
        return first == "PASS", verdict[:200]
    return judge


def run_eval(client: Ollama, model_tag: str, cases: list[dict], *, options: dict,
             judge_model: str = "", think=None, on_case=None) -> dict:
    judge_fn = make_judge(client, judge_model) if judge_model else None
    results, passed, latencies = [], 0, []
    for c in cases:
        msgs = c.get("messages") or [{"role": "user", "content": c["prompt"]}]
        t0 = time.monotonic()
        try:
            r = client.chat(model_tag, msgs, options=options, think=think)
            answer = strip_thinking(r.get("message", {}).get("content", ""))
            error = ""
        except OllamaError as e:
            answer, error = "", e.message
        ms = int((time.monotonic() - t0) * 1000)
        latencies.append(ms)
        checks = run_checks(c, answer, judge_fn) if not error else [
            {"check": "request", "pass": False, "detail": error}]
        ok = all(ch["pass"] for ch in checks)
        passed += ok
        row = {"id": c["id"], "prompt": c.get("prompt") or msgs[-1]["content"], "answer": answer,
               "pass": ok, "checks": checks, "latency_ms": ms}
        results.append(row)
        if on_case:
            on_case(row)
    total = len(cases)
    return {"model": model_tag, "score": round(passed / total, 4) if total else 0.0,
            "passed": passed, "total": total,
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
            "options": options, "judge": judge_model or None, "cases": results}
