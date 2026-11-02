"""Text helpers: normalization, counting and extracting code/JSON/numbers from model responses."""
from __future__ import annotations

import json
import re
import unicodedata

_QUOTES = {"«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'"}
_DASHES = {"—": "-", "–": "-", "−": "-"}
_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_NUMBER = re.compile(r"-?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?")


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> blocks that some reasoning models put into the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()


def normalize(text: str) -> str:
    """Lowercase, ё→е, unify quotes, dashes and whitespace."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in {**_QUOTES, **_DASHES}.items():
        text = text.replace(src, dst)
    text = text.replace("ё", "е").replace("Ё", "Е").lower()
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def script_ratio(text: str, script: str = "cyrillic") -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if script == "cyrillic":
        hits = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
    elif script == "latin":
        hits = sum(1 for c in letters if c.isascii())
    else:
        raise ValueError(f"unknown script {script}")
    return hits / len(letters)


def bullets(text: str) -> list[str]:
    pattern = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s+(.+)$")
    return [m.group(1) for line in text.splitlines() if (m := pattern.match(line))]


def sentence_count(text: str) -> int:
    body = " ".join(text.split())
    # ignore decimals like 3.5 and common abbreviations
    body = re.sub(r"(\d)[.,](\d)", r"\1\2", body)
    body = re.sub(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Dr|St|No)\.", r"\1", body, flags=re.I)
    parts = [p for p in re.split(r"[.!?]+(?:\s+|$)", body) if p.strip()]
    return len(parts)


def parse_number(text: str) -> float | None:
    """First number in text; understands '1 608', '1608,00', '3.12'."""
    m = _NUMBER.search(text)
    if not m:
        return None
    raw = re.sub(r"[   ]", "", m.group(0)).replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def all_numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER.finditer(text):
        raw = re.sub(r"[   ]", "", m.group(0)).replace(",", ".")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def extract_code(text: str, required_name: str | None = None) -> str | None:
    """Pick the Python code from a response: fenced block containing the function, else best guess."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.S | re.I)
    with_defs = [b for b in blocks if re.search(r"^\s*def\s", b, flags=re.M)]
    if required_name and not any(re.search(rf"def\s+{re.escape(required_name)}\s*\(", b) for b in with_defs):
        with_defs = []
    if with_defs:
        # functions may be split across blocks; later definitions override earlier ones ("before/after")
        return "\n\n".join(with_defs)
    if blocks:
        return max(blocks, key=len)
    if re.search(r"^\s*(def|import|from)\s", text, flags=re.M):
        return text
    return None


def extract_json(text: str):
    """Parse JSON from a response: fenced ```json block, the whole text, or the first balanced {...}/[...]."""
    candidates = re.findall(r"```(?:json)?[ \t]*\n(.*?)```", text, flags=re.S | re.I)
    candidates.append(text)
    for cand in candidates:
        try:
            return json.loads(cand.strip())
        except (json.JSONDecodeError, ValueError):
            pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break
            start = text.find(opener, start + 1)
    raise ValueError("no valid JSON found in response")
