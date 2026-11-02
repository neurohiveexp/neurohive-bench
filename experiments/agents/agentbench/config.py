"""Task and agent configs: experiments/agents/tasks.json and agents.json."""
import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_TASKS = os.path.join(HERE, "tasks.json")
DEFAULT_AGENTS = os.path.join(HERE, "agents.json")
DEFAULT_RESULTS = os.path.join(REPO_ROOT, "results", "agents")

TASK_TYPES = ("feature", "bugfix", "greenfield")


class ConfigError(Exception):
    pass


@dataclass
class Task:
    id: str
    type: str
    repo: str | None
    commit: str | None
    prompt: str
    test_cmd: str
    hidden_tests: list = field(default_factory=list)
    setup_cmd: str | None = None
    agent_timeout_s: float = 1800
    test_timeout_s: float = 600
    meta: dict = field(default_factory=dict)


@dataclass
class Agent:
    id: str
    title: str
    cmd: list
    parser: str = "generic"
    model: str | None = None
    model_args: list = field(default_factory=list)
    version_cmd: list | None = None
    install: str = ""
    env: dict = field(default_factory=dict)
    stdin: bool = False
    price: dict | None = None


def _resolve(base_dir, path):
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_tasks(path=DEFAULT_TASKS):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    tasks, seen = [], set()
    for raw in data.get("tasks", []):
        tid = raw.get("id")
        if not tid:
            raise ConfigError(f"{path}: task without id")
        if tid in seen:
            raise ConfigError(f"{path}: duplicate task id {tid!r}")
        seen.add(tid)
        ttype = raw.get("type")
        if ttype not in TASK_TYPES:
            raise ConfigError(f"task {tid}: type must be one of {TASK_TYPES}, got {ttype!r}")
        repo, commit = raw.get("repo"), raw.get("commit")
        if ttype == "greenfield":
            if repo:
                raise ConfigError(f"task {tid}: greenfield task starts from an empty repo, remove 'repo'")
        elif not repo or not commit:
            raise ConfigError(f"task {tid}: {ttype} task needs 'repo' and 'commit'")
        if repo and not repo.startswith(("http://", "https://", "git@", "file://", "ssh://")):
            repo = os.path.abspath(_resolve(base, repo))
        if "prompt" in raw:
            prompt = raw["prompt"]
        elif "prompt_file" in raw:
            with open(_resolve(base, raw["prompt_file"]), encoding="utf-8") as f:
                prompt = f.read()
        else:
            raise ConfigError(f"task {tid}: needs 'prompt' or 'prompt_file'")
        if not raw.get("test_cmd"):
            raise ConfigError(f"task {tid}: needs 'test_cmd'")
        hidden = []
        for h in raw.get("hidden_tests", []):
            if "path" not in h or ("from_commit" in h) == ("file" in h):
                raise ConfigError(f"task {tid}: hidden test needs 'path' and one of 'from_commit' / 'file'")
            if "from_commit" in h and not repo:
                raise ConfigError(f"task {tid}: 'from_commit' hidden tests need a repo")
            h = dict(h)
            if "file" in h:
                h["file"] = _resolve(base, h["file"])
                if not os.path.isfile(h["file"]):
                    raise ConfigError(f"task {tid}: hidden test file not found: {h['file']}")
            hidden.append(h)
        known = {"id", "type", "repo", "commit", "prompt", "prompt_file", "test_cmd", "hidden_tests",
                 "setup_cmd", "agent_timeout_s", "test_timeout_s"}
        tasks.append(Task(
            id=tid, type=ttype, repo=repo, commit=commit, prompt=prompt.strip() + "\n",
            test_cmd=raw["test_cmd"], hidden_tests=hidden, setup_cmd=raw.get("setup_cmd"),
            agent_timeout_s=float(raw.get("agent_timeout_s", 1800)),
            test_timeout_s=float(raw.get("test_timeout_s", 600)),
            meta={k: v for k, v in raw.items() if k not in known},
        ))
    return tasks


def load_agents(path=DEFAULT_AGENTS):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    agents = {}
    for aid, raw in data.items():
        if aid.startswith("_"):
            continue
        cmd = raw.get("cmd")
        if not cmd or not isinstance(cmd, list):
            raise ConfigError(f"agent {aid}: 'cmd' must be a non-empty list")
        stdin = bool(raw.get("stdin", False))
        if not stdin and not any("{prompt}" in part for part in cmd):
            raise ConfigError(f"agent {aid}: 'cmd' must contain {{prompt}} or set \"stdin\": true")
        agents[aid] = Agent(
            id=aid, title=raw.get("title", aid), cmd=cmd, parser=raw.get("parser", "generic"),
            model=raw.get("model"), model_args=raw.get("model_args", []),
            version_cmd=raw.get("version_cmd"), install=raw.get("install", ""),
            env=raw.get("env", {}), stdin=stdin, price=raw.get("price"),
        )
    return agents


def pick(items, ids, kind):
    """Select items by comma-separated ids, keeping the requested order."""
    if not ids:
        return list(items.values()) if isinstance(items, dict) else list(items)
    index = items if isinstance(items, dict) else {i.id: i for i in items}
    wanted = [i.strip() for i in ids.split(",") if i.strip()]
    unknown = [i for i in wanted if i not in index]
    if unknown:
        raise ConfigError(f"unknown {kind}: {', '.join(unknown)}; known: {', '.join(index)}")
    return [index[i] for i in wanted]
