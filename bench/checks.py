"""Deterministic checks. Every rule returns (passed, detail); a task passes when all its rules pass."""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import text as T


@dataclass
class RuleResult:
    rule: str
    passed: bool
    detail: str = ""


@dataclass
class CheckResult:
    task_id: str
    passed: bool
    rules: list[RuleResult] = field(default_factory=list)

    def summary(self) -> str:
        failed = [r for r in self.rules if not r.passed]
        if not failed:
            return "pass"
        return "; ".join(f"{r.rule}: {r.detail}" for r in failed)


# ---------- text rules ----------

def _word_count(resp, rule, task):
    n = T.word_count(resp)
    lo, hi = rule.get("min", 0), rule.get("max", 10**9)
    return lo <= n <= hi, f"{n} words (allowed {lo}–{hi if hi < 10**9 else '∞'})"


def _bullets(resp, rule, task):
    n = len(T.bullets(resp))
    return n == rule["exact"], f"{n} bullet points (need {rule['exact']})"


def _sentences(resp, rule, task):
    n = T.sentence_count(resp)
    return n <= rule["max"], f"{n} sentences (max {rule['max']})"


def _script(resp, rule, task):
    ratio = T.script_ratio(resp, rule["script"])
    return ratio >= rule["min_ratio"], f"{rule['script']} share {ratio:.2f} (min {rule['min_ratio']})"


def _no_cyrillic(resp, rule, task):
    found = re.findall(r"[Ѐ-ӿ]+", resp)
    return not found, "no Cyrillic" if not found else f"Cyrillic left: {' '.join(found[:5])}"


def _regex(resp, rule, task):
    ok = re.search(rule["pattern"], resp) is not None
    return ok, rule.get("name", rule["pattern"]) + (" found" if ok else " missing")


def _facts(resp, rule, task):
    hits, misses = [], []
    for group in rule["groups"]:
        if any(re.search(p, resp, flags=re.I) for p in group["any"]):
            hits.append(group["name"])
        else:
            misses.append(group["name"])
    need = rule.get("min_groups", len(rule["groups"]))
    ok = len(hits) >= need
    detail = f"{len(hits)}/{len(rule['groups'])} facts (need {need})"
    if misses:
        detail += "; missing: " + ", ".join(misses)
    return ok, detail


def _fixes(resp, rule, task):
    norm = T.normalize(resp)
    problems = []
    for item in rule["items"]:
        bad = re.search(item["bad"], norm)
        good = re.search(item["good"], norm)
        if bad or not good:
            problems.append(item["name"])
    ok = not problems
    return ok, "all fixes made" if ok else "not fixed: " + ", ".join(problems)


def _similarity(resp, rule, task):
    ref = (task.path / rule["reference"]).read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, T.normalize(ref), T.normalize(resp)).ratio()
    return ratio >= rule["min"], f"similarity to reference {ratio:.3f} (min {rule['min']})"


def _numbered_answers(resp, rule, task):
    lines = {}
    for line in resp.splitlines():
        m = re.match(r"^\s*\**\s*(\d+)\s*[).:]\**\s*(.*)$", line)
        if m and int(m.group(1)) not in lines:
            lines[int(m.group(1))] = m.group(2)
    problems = []
    for ans in rule["answers"]:
        got = lines.get(ans["n"])
        if got is None:
            problems.append(f"answer {ans['n']} missing")
            continue
        if "number" in ans:
            # "200 × 6,48 + 100 × 3,12 = 1608" → take what follows the last "="
            num = T.parse_number(got.rsplit("=", 1)[-1]) if "=" in got else T.parse_number(got)
            if num is None or abs(num - ans["number"]) > ans.get("tolerance", 0):
                problems.append(f"answer {ans['n']}: got {got!r}, expected {ans['number']}")
        if "text" in ans and T.normalize(ans["text"]) not in T.normalize(got):
            problems.append(f"answer {ans['n']}: got {got!r}, expected {ans['text']!r}")
    return not problems, "all answers correct" if not problems else "; ".join(problems)


def _final_answer(resp, rule, task):
    prefix = rule["prefix"]
    lines = [ln for ln in resp.splitlines() if prefix.lower() in ln.lower()]
    if not lines:
        return False, f"no line with {prefix!r}"
    tail = lines[-1][lines[-1].lower().rindex(prefix.lower()) + len(prefix):]
    nums = T.all_numbers(tail)
    expected = rule["values"]
    if len(nums) < len(expected):
        return False, f"expected {len(expected)} numbers after {prefix!r}, got {tail.strip()!r}"
    tol = rule.get("tolerance", 0)
    ok = all(abs(g - e) <= tol for g, e in zip(nums, expected))
    return ok, f"got {nums[:len(expected)]}, expected {expected}"


# ---------- JSON rules ----------

def _field_ok(kind: str, got, exp) -> bool:
    if kind == "exact":
        return got == exp
    if kind == "exact_ci":
        return isinstance(got, str) and got.strip().lower() == str(exp).lower()
    if kind == "contains_ci":
        core = re.sub(r"[\W_]", "", T.normalize(str(exp)))
        return isinstance(got, str) and core in re.sub(r"[\W_]", "", T.normalize(got))
    if kind == "alnum_ci":
        def key(v):
            return re.sub(r"[^0-9a-zа-я]", "", T.normalize(str(v)))
        return isinstance(got, str) and key(got) == key(exp)
    if kind == "number":
        if exp is None:
            return got is None
        try:
            return got is not None and not isinstance(got, bool) and abs(float(got) - float(exp)) <= 0.01
        except (TypeError, ValueError):
            return False
    if kind == "spdx":
        def key(v):
            return re.sub(r"[^0-9a-z]", "", str(v).lower()).replace("license", "")
        return isinstance(got, str) and key(got) == key(exp)
    raise ValueError(f"unknown field check {kind}")


def _json_object(resp, rule, task):
    expected = json.loads((task.path / rule["expected"]).read_text(encoding="utf-8"))
    try:
        got = T.extract_json(resp)
    except ValueError as e:
        return False, str(e)
    if not isinstance(got, dict):
        return False, f"expected a JSON object, got {type(got).__name__}"
    bad = [f"{k}: got {got.get(k)!r}, expected {expected[k]!r}"
           for k, kind in rule["fields"].items() if not _field_ok(kind, got.get(k), expected[k])]
    return not bad, "all fields correct" if not bad else "; ".join(bad)


def _json_records(resp, rule, task):
    expected = json.loads((task.path / rule["expected"]).read_text(encoding="utf-8"))
    try:
        got = T.extract_json(resp)
    except ValueError as e:
        return False, str(e)
    if isinstance(got, dict):  # tolerate {"receipts": [...]}
        lists = [v for v in got.values() if isinstance(v, list)]
        got = lists[0] if len(lists) == 1 else got
    if not isinstance(got, list):
        return False, "expected a JSON array"
    if len(got) != len(expected):
        return False, f"{len(got)} records, expected {len(expected)}"
    bad = []
    for i, (g, e) in enumerate(zip(got, expected), start=1):
        if not isinstance(g, dict):
            bad.append(f"record {i} is not an object")
            continue
        for k, kind in rule["fields"].items():
            if not _field_ok(kind, g.get(k), e[k]):
                bad.append(f"record {i} {k}: got {g.get(k)!r}, expected {e[k]!r}")
    return not bad, "all records correct" if not bad else "; ".join(bad)


# ---------- code rules ----------

def _python_tests(resp, rule, task):
    code = T.extract_code(resp, rule.get("function"))
    if code is None:
        return False, "no Python code found"
    with tempfile.TemporaryDirectory(prefix="nhbench_") as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        shutil.copy(task.path / rule["tests"], Path(tmp, "tests.py"))
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "tests.py"], cwd=tmp, env=env,
                capture_output=True, text=True, timeout=rule.get("timeout", 20),
            )
        except subprocess.TimeoutExpired:
            return False, "tests timed out"
    out = (proc.stderr or proc.stdout).strip().splitlines()
    summary = next((ln for ln in reversed(out) if ln.startswith(("OK", "FAILED", "Ran"))), out[-1] if out else "")
    if proc.returncode == 0:
        return True, summary or "tests passed"
    first_error = next((ln for ln in out if ln.startswith(("FAIL:", "ERROR:", "ImportError", "SyntaxError", "NameError"))), "")
    return False, f"{summary} {first_error}".strip()


RULES = {
    "word_count": _word_count,
    "bullets": _bullets,
    "sentences": _sentences,
    "script": _script,
    "no_cyrillic": _no_cyrillic,
    "regex": _regex,
    "facts": _facts,
    "fixes": _fixes,
    "similarity": _similarity,
    "numbered_answers": _numbered_answers,
    "final_answer": _final_answer,
    "json_object": _json_object,
    "json_records": _json_records,
    "python_tests": _python_tests,
}


def check(task, response: str) -> CheckResult:
    response = T.strip_reasoning(response or "")
    results = []
    for rule in task.check["rules"]:
        fn = RULES[rule["type"]]
        try:
            ok, detail = fn(response, rule, task)
        except Exception as e:  # a broken response must never crash the run
            ok, detail = False, f"checker error: {type(e).__name__}: {e}"
        results.append(RuleResult(rule["type"], bool(ok), detail))
    return CheckResult(task.id, all(r.passed for r in results), results)
