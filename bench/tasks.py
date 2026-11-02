"""Loading tasks from tasks/<id>/{task.json, prompt.md, reference.*}."""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


@dataclass
class Task:
    id: str
    version: int
    title: str
    category: str
    language: str
    prompt: str
    check: dict
    path: Path
    skills: tuple = ()
    notes: str = ""

    def reference(self) -> str:
        for name in ("reference.md", "reference.py", "reference.json"):
            p = self.path / name
            if p.exists():
                return p.read_text(encoding="utf-8")
        raise FileNotFoundError(f"no reference for {self.id}")


def load_tasks(patterns: list[str] | None = None, tasks_dir: Path = TASKS_DIR) -> list[Task]:
    tasks = []
    for meta_path in sorted(tasks_dir.glob("*/task.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if patterns and not any(fnmatch.fnmatch(meta["id"], p) for p in patterns):
            continue
        tasks.append(Task(
            id=meta["id"],
            version=meta["version"],
            title=meta["title"],
            category=meta["category"],
            language=meta["language"],
            prompt=(meta_path.parent / "prompt.md").read_text(encoding="utf-8").strip(),
            check=meta["check"],
            path=meta_path.parent,
            skills=tuple(meta.get("skills", ())),
            notes=meta.get("notes", ""),
        ))
    if patterns and not tasks:
        raise SystemExit(f"no tasks match {patterns}")
    return tasks
