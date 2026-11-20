"""Training data: load, normalize, validate.

Everything is normalized to the chat format that Unsloth/TRL/MLX all accept:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

Also accepted on input (converted automatically):
    {"conversations": [{"from": "human"|"gpt", "value": "..."}]}   (ShareGPT)
    {"prompt": "...", "response": "..."}   (or "completion"/"output")
    {"instruction": "...", "input": "...", "output": "..."}   (Alpaca)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROLE_MAP = {
    "system": "system", "user": "user", "human": "user",
    "assistant": "assistant", "gpt": "assistant", "model": "assistant", "bot": "assistant",
}


def normalize(obj: dict) -> list[dict]:
    """Return a clean messages list or raise ValueError with a readable reason."""
    if not isinstance(obj, dict):
        raise ValueError("each line must be a JSON object")
    if "messages" in obj or "conversations" in obj:
        raw = obj.get("messages", obj.get("conversations"))
        if not isinstance(raw, list):
            raise ValueError('"messages" must be a list')
        msgs = []
        for m in raw:
            if not isinstance(m, dict):
                raise ValueError("each message must be an object")
            role = m.get("role", m.get("from"))
            content = m.get("content", m.get("value"))
            if role is None or content is None:
                raise ValueError("each message needs role/content (or from/value)")
            mapped = ROLE_MAP.get(str(role).lower())
            if not mapped:
                raise ValueError(f"unknown role {role!r}")
            msgs.append({"role": mapped, "content": str(content)})
    elif "prompt" in obj and any(k in obj for k in ("response", "completion", "output")):
        answer = obj.get("response", obj.get("completion", obj.get("output")))
        msgs = [{"role": "user", "content": str(obj["prompt"])},
                {"role": "assistant", "content": str(answer)}]
        if obj.get("system"):
            msgs.insert(0, {"role": "system", "content": str(obj["system"])})
    elif "instruction" in obj and "output" in obj:
        user = str(obj["instruction"])
        if obj.get("input"):
            user += "\n\n" + str(obj["input"])
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": str(obj["output"])}]
    else:
        raise ValueError("unrecognized format (expected messages, conversations, "
                         "prompt+response, or instruction+output)")

    for i, m in enumerate(msgs):
        if not m["content"].strip():
            raise ValueError(f"message {i + 1} ({m['role']}) is empty")
        if m["role"] == "system" and i != 0:
            raise ValueError("a system message is only allowed as the first message")
    roles = [m["role"] for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("needs at least one user and one assistant message")
    if roles[-1] != "assistant":
        raise ValueError("the last message should be the assistant's reply (that's what gets learned)")
    return msgs


def load(path: Path) -> tuple[list[list[dict]], list[tuple[int, str]]]:
    """Load a JSONL file. Returns (examples, errors) where errors are (line_no, reason)."""
    examples, errors = [], []
    if not path.exists():
        return [], [(0, f"file not found: {path}")]
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                examples.append(normalize(json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append((no, f"invalid JSON ({e.msg})"))
            except ValueError as e:
                errors.append((no, str(e)))
    return examples, errors


def est_tokens(msgs: list[dict]) -> int:
    # Rough rule of thumb (~4 characters per token for English) plus template overhead.
    return sum(len(m["content"]) for m in msgs) // 4 + 8 * len(msgs)


def check(path: Path, max_seq_length: int = 2048) -> dict:
    examples, errors = load(path)
    seen, dupes = set(), 0
    lengths = []
    for ex in examples:
        key = json.dumps(ex, sort_keys=True)
        if key in seen:
            dupes += 1
        seen.add(key)
        lengths.append(est_tokens(ex))
    warnings = []
    if dupes:
        warnings.append(f"{dupes} duplicate example(s)")
    too_long = sum(1 for n in lengths if n > max_seq_length)
    if too_long:
        warnings.append(f"{too_long} example(s) look longer than max_seq_length={max_seq_length} "
                        "and will be truncated")
    if 0 < len(examples) < 50:
        warnings.append("fewer than 50 examples: fine-tuning this small rarely beats a good "
                        "system prompt. Aim for 100+ high-quality examples.")
    return {
        "path": str(path),
        "count": len(examples),
        "errors": errors,
        "warnings": warnings,
        "with_system": sum(1 for ex in examples if ex[0]["role"] == "system"),
        "avg_tokens": int(sum(lengths) / len(lengths)) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
    }


def prepare(src: Path, dest: Path, system_prompt: str | None = None) -> dict:
    """Write a normalized copy for training. Adds the model's system prompt to examples that
    lack one, so the model learns in the same context it will be served in."""
    examples, errors = load(src)
    if errors:
        line, why = errors[0]
        raise ValueError(f"{src}: line {line}: {why} ({len(errors)} problem(s) total; "
                         f"run `rack data check` for the full list)")
    if not examples:
        raise ValueError(f"{src} has no examples")
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(dest, "w", encoding="utf-8") as f:
        for msgs in examples:
            if system_prompt and msgs[0]["role"] != "system":
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            line = json.dumps({"messages": msgs}, ensure_ascii=False)
            h.update(line.encode("utf-8"))
            f.write(line + "\n")
    return {"count": len(examples), "sha256": h.hexdigest()}
