"""Parse Ollama Modelfiles (enough of the format to import fine-tuned models).

Reference: https://docs.ollama.com/modelfile
"""

from __future__ import annotations

MULTI_VALUE_PARAMS = {"stop"}


def _read_value(text: str, pos: int) -> tuple[str, int]:
    """Read a value starting at pos: triple-quoted, double-quoted, or rest of line."""
    n = len(text)
    while pos < n and text[pos] in " \t":
        pos += 1
    if text.startswith('"""', pos):
        end = text.find('"""', pos + 3)
        if end == -1:
            raise ValueError('Unterminated """ block in Modelfile')
        return text[pos + 3:end], end + 3
    if pos < n and text[pos] == '"':
        out, i = [], pos + 1
        while i < n and text[i] != '"':
            if text[i] == "\\" and i + 1 < n:
                i += 1
                out.append({"n": "\n", "t": "\t"}.get(text[i], text[i]))
            else:
                out.append(text[i])
            i += 1
        return "".join(out), i + 1
    end = text.find("\n", pos)
    end = n if end == -1 else end
    return text[pos:end].strip(), end


def _read_word(text: str, pos: int) -> tuple[str, int]:
    n = len(text)
    while pos < n and text[pos] in " \t":
        pos += 1
    start = pos
    while pos < n and not text[pos].isspace():
        pos += 1
    return text[start:pos], pos


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse(text: str) -> dict:
    out: dict = {"from": None, "template": None, "system": None, "license": [],
                 "parameters": {}, "messages": [], "adapters": []}
    pos, n = 0, len(text)
    while pos < n:
        # skip blank space and comments
        while pos < n and text[pos] in " \t\r\n":
            pos += 1
        if pos >= n:
            break
        if text[pos] == "#":
            nl = text.find("\n", pos)
            pos = n if nl == -1 else nl + 1
            continue
        word, pos = _read_word(text, pos)
        kw = word.upper()
        if kw == "PARAMETER":
            key, pos = _read_word(text, pos)
            val, pos = _read_value(text, pos)
            key = key.lower()
            if key in MULTI_VALUE_PARAMS:
                out["parameters"].setdefault(key, []).append(val)
            else:
                out["parameters"][key] = _coerce(val)
        elif kw == "MESSAGE":
            role, pos = _read_word(text, pos)
            content, pos = _read_value(text, pos)
            out["messages"].append({"role": role.lower(), "content": content})
        elif kw in ("FROM", "TEMPLATE", "SYSTEM", "ADAPTER", "LICENSE", "REQUIRES", "RENDERER", "PARSER"):
            val, pos = _read_value(text, pos)
            if kw == "LICENSE":
                out["license"].append(val)
            elif kw == "ADAPTER":
                out["adapters"].append(val)
            else:
                out[kw.lower()] = val
        else:
            # Unknown instruction: skip the rest of the line rather than guessing.
            nl = text.find("\n", pos)
            pos = n if nl == -1 else nl + 1
    return out


# Stop tokens for well-known chat formats, used when a fine-tune's Modelfile doesn't set any.
# They match what the official Ollama library models use for these families.
_FAMILY_STOPS = [
    ("<|im_start|>", ["<|im_end|>", "<|im_start|>"]),                       # ChatML: Qwen, many others
    ("<|start_header_id|>", ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>"]),  # Llama 3
    ("<start_of_turn>", ["<end_of_turn>"]),                                 # Gemma 2/3
]


def infer_stops(template: str | None) -> list[str]:
    if not template:
        return []
    for marker, stops in _FAMILY_STOPS:
        if marker in template:
            return list(stops)
    return []
