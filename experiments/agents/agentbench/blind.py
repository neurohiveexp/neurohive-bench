"""Blind export: diffs under letters A, B, C, D so reviewers do not know which agent wrote them."""
import csv
import json
import os
import random
import re
import string

from .results import load_records

# Files that give the agent away; dropped from the diff together with their hunks.
AGENT_FILES = re.compile(
    r"(^|/)(CLAUDE\.md|CLAUDE\.local\.md|AGENTS\.md|AGENT\.md|KIMI\.md|\.claude|\.codex|\.opencode|"
    r"opencode\.jsonc?|\.kimi|\.aider[^/]*|\.cursor[^/]*|\.github/copilot[^/]*)(/|$)")
AGENT_WORDS = ["claude code", "claude", "anthropic", "codex", "openai", "opencode", "kimi code",
               "kimi", "moonshot", "sonnet", "opus", "haiku", "gpt-[0-9][\\w.-]*", "o[0-9]-[\\w-]+",
               "co-authored-by:[^\\n]*", "generated with[^\\n]*"]
REDACTED = "[агент]"


def split_files(diff):
    """[(path, text)] per `diff --git` section."""
    parts = re.split(r"(?m)^(?=diff --git )", diff)
    out = []
    for part in parts:
        if not part.strip():
            continue
        m = re.match(r"diff --git a/(.*?) b/(.*)$", part.splitlines()[0])
        out.append((m.group(2) if m else "", part))
    return out


def sanitize(diff, extra_words=()):
    """Drop agent-specific files, replace agent and vendor names. Returns (diff, dropped, replaced)."""
    kept, dropped = [], []
    for path, text in split_files(diff):
        (dropped if AGENT_FILES.search(path) else kept).append((path, text))
    text = "".join(t for _, t in kept)
    words = sorted({*AGENT_WORDS, *(re.escape(w) for w in extra_words if len(w) > 2)}, key=len, reverse=True)
    pattern = re.compile(r"(?i)(?<![\w-])(" + "|".join(words) + r")(?![\w])")
    text, replaced = pattern.subn(REDACTED, text)
    return text, [p for p, _ in dropped], replaced


def letters(n):
    if n > len(string.ascii_uppercase):
        raise ValueError("more than 26 agents")
    return list(string.ascii_uppercase[:n])


def export(results_paths, out_dir, *, seed, run=1, key_path=None, tasks=None, results_dir=None,
           redact=True):
    """Write out_dir/<task>/<letter>.diff, out_dir/<task>/task.md, out_dir/grading.csv and a separate key.

    Letters are shuffled per task with `seed`, so one letter is not the same agent on every task.
    Only `run` number `run` of each (task, agent) is exported, so every agent gets the same number of samples."""
    records = load_records(results_paths)
    chosen = {}
    for r in records:
        if r["run"] == run and r.get("logs", {}).get("changes"):
            chosen.setdefault((r["task_id"], r["agent"]), r)
    if not chosen:
        raise ValueError(f"no runs with number {run} and a saved diff in {', '.join(results_paths)}")
    by_task = {}
    for (task_id, agent), r in sorted(chosen.items()):
        by_task.setdefault(task_id, []).append(r)
    agents_seen = sorted({a for _, a in chosen})
    extra = agents_seen + [r.get("agent_title") or "" for r in chosen.values()]
    os.makedirs(out_dir, exist_ok=True)
    key = {"seed": seed, "run": run, "tasks": {}}
    grading_rows = []
    task_prompts = {t.id: t.prompt for t in (tasks or [])}
    for task_id, recs in by_task.items():
        rng = random.Random(f"{seed}:{task_id}")
        order = recs[:]
        rng.shuffle(order)
        tdir = os.path.join(out_dir, task_id)
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "task.md"), "w", encoding="utf-8") as f:
            f.write(f"# {task_id}\n\n{task_prompts.get(task_id, '(текст задания: tasks.json)').strip()}\n")
        key["tasks"][task_id] = {}
        for letter, r in zip(letters(len(order)), order):
            base = results_dir or os.path.dirname(os.path.abspath(r["_source"]))
            with open(os.path.join(base, r["logs"]["changes"]), encoding="utf-8") as f:
                diff = f.read()
            dropped, replaced = [], 0
            if redact:
                diff, dropped, replaced = sanitize(diff, extra)
            with open(os.path.join(tdir, f"{letter}.diff"), "w", encoding="utf-8") as f:
                f.write(diff)
            key["tasks"][task_id][letter] = {"agent": r["agent"], "run_id": r["run_id"],
                                             "passed": r["passed"], "dropped_files": dropped,
                                             "redactions": replaced}
            grading_rows.append([task_id, letter, "", "", "", ""])
    with open(os.path.join(out_dir, "grading.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "letter", "correctness_1_5", "readability_1_5", "tests_1_5", "comment"])
        w.writerows(grading_rows)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Слепая оценка кода\n\n"
            "В каждой папке — текст задания (`task.md`) и изменения нескольких участников под буквами A, B, C, D. "
            "Буквы перемешаны отдельно для каждой задачи. Названия агентов и моделей заменены на «[агент]», "
            "служебные файлы агентов (CLAUDE.md, AGENTS.md и т. п.) убраны.\n\n"
            "Заполните `grading.csv`: оценки 1–5 за корректность, читаемость и тесты, комментарий по желанию.\n")
    key_path = key_path or os.path.abspath(out_dir.rstrip(os.sep)) + ".key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    return key_path, key
