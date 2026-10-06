import json
from pathlib import Path


def _scan(text: str, drop_comments: bool) -> str:
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif drop_comments and ch == "/" and text[i + 1:i + 2] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif not drop_comments and ch == ",":
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j] in "]}":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_dataset(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    text = _scan(_scan(text, drop_comments=True), drop_comments=False).strip()

    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = []
        for no, line in enumerate(text.splitlines(), 1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {no}, col {e.colno}: {e.msg}") from e

    return [r for r in rows if not r.get("skip")]