"""Running agent CLIs non-interactively and reading tokens / cost from their output."""
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass

from .config import REPO_ROOT


class AgentNotFound(Exception):
    pass


def check_installed(agent):
    exe = agent.cmd[0]
    path = shutil.which(exe)
    if path is None:
        hint = f"\n  установка: {agent.install}" if agent.install else ""
        raise AgentNotFound(
            f"агент {agent.id} ({agent.title}): команда {exe!r} не найдена в PATH.{hint}\n"
            f"  команды запуска описаны в experiments/agents/README.md; "
            f"другой путь к CLI можно указать в agents.json")
    return path


def agent_version(agent):
    if not agent.version_cmd:
        return None
    try:
        p = subprocess.run(agent.version_cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (p.stdout or p.stderr).strip().splitlines()
    return out[0][:200] if out else None


def build_command(agent, prompt, model=None):
    model = model or agent.model
    cmd = [part.replace("{prompt}", prompt) for part in agent.cmd]
    if model:
        cmd += [part.replace("{model}", model) for part in agent.model_args]
    return cmd


@dataclass
class ProcResult:
    exit_code: int | None
    duration_s: float
    timed_out: bool


def run_process(cmd, cwd, stdout_path, stderr_path, timeout, env=None, stdin_text=None):
    """Run `cmd` in its own process group; kill the whole group on timeout."""
    full_env = {**os.environ, **(env or {})}
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        start = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, env=full_env,
                                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                start_new_session=True)
        timed_out = False
        try:
            if stdin_text is not None:
                proc.communicate(stdin_text.encode(), timeout=timeout)
            else:
                proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
        duration = time.monotonic() - start
    return ProcResult(None if timed_out else proc.returncode, round(duration, 3), timed_out)


def _kill_group(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            continue


def run_shell(command, cwd, log_path, timeout, env=None):
    """Setup and test commands: a shell string, stdout+stderr in one log."""
    with open(log_path, "wb") as log:
        start = time.monotonic()
        proc = subprocess.Popen(command, shell=True, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})},
                                start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
        duration = time.monotonic() - start
    return ProcResult(None if timed_out else proc.returncode, round(duration, 3), timed_out)


# ---------------------------------------------------------------- parsing

def iter_json(text):
    """JSON values in agent output: the whole text, or one value per line (JSONL)."""
    text = text.strip()
    if not text:
        return []
    try:
        return [json.loads(text)]
    except ValueError:
        pass
    values = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                values.append(json.loads(line))
            except ValueError:
                continue
    return values


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _add(a, b):
    if b is None:
        return a
    return b if a is None else a + b


def empty_usage():
    return {"input_tokens": None, "output_tokens": None, "cached_tokens": None,
            "cache_write_tokens": None, "cost_usd": None, "turns": None}


def parse_claude(values):
    """`claude -p --output-format json` (or stream-json): the final `result` object has the totals.
    Anthropic's input_tokens excludes cache reads and writes, so they are added back."""
    u = empty_usage()
    result = next((v for v in reversed(values) if isinstance(v, dict) and v.get("type") == "result"), None)
    if result is None:
        return u
    usage = result.get("usage") or {}
    inp = _num(usage.get("input_tokens"))
    read = _num(usage.get("cache_read_input_tokens")) or 0
    write = _num(usage.get("cache_creation_input_tokens")) or 0
    if inp is not None:
        u["input_tokens"] = inp + read + write
        u["cached_tokens"] = read
        u["cache_write_tokens"] = write
    u["output_tokens"] = _num(usage.get("output_tokens"))
    u["cost_usd"] = _num(result.get("total_cost_usd", result.get("cost_usd")))
    u["turns"] = _num(result.get("num_turns"))
    return u


def parse_codex(values):
    """`codex exec --json`: sum `turn.completed` usage (input_tokens includes cached_input_tokens).
    Older versions print `token_count` events with running totals; the last one is used."""
    u = empty_usage()
    turns = 0
    for v in values:
        if isinstance(v, dict) and v.get("type") == "turn.completed" and isinstance(v.get("usage"), dict):
            usage = v["usage"]
            turns += 1
            u["input_tokens"] = _add(u["input_tokens"], _num(usage.get("input_tokens")))
            u["cached_tokens"] = _add(u["cached_tokens"], _num(usage.get("cached_input_tokens")))
            u["output_tokens"] = _add(u["output_tokens"], _num(usage.get("output_tokens")))
    if turns:
        u["turns"] = turns
        return u
    for v in reversed(values):
        msg = v.get("msg", v) if isinstance(v, dict) else None
        if isinstance(msg, dict) and msg.get("type") == "token_count":
            total = (msg.get("info") or {}).get("total_token_usage") or {}
            u["input_tokens"] = _num(total.get("input_tokens"))
            u["cached_tokens"] = _num(total.get("cached_input_tokens"))
            u["output_tokens"] = _num(total.get("output_tokens"))
            break
    return u


def parse_opencode(values):
    """`opencode run --format json`: sum tokens and cost of `step_finish` parts.
    tokens.input excludes cache reads and writes, so they are added back."""
    u = empty_usage()
    steps = 0
    for v in values:
        if not isinstance(v, dict) or v.get("type") != "step_finish":
            continue
        part = v.get("part") if isinstance(v.get("part"), dict) else v
        tokens = part.get("tokens") or {}
        cache = tokens.get("cache") or {}
        steps += 1
        read, write = _num(cache.get("read")) or 0, _num(cache.get("write")) or 0
        inp = _num(tokens.get("input"))
        if inp is not None:
            u["input_tokens"] = _add(u["input_tokens"], inp + read + write)
            u["cached_tokens"] = _add(u["cached_tokens"], read)
            u["cache_write_tokens"] = _add(u["cache_write_tokens"], write)
        out = _num(tokens.get("output"))
        if out is not None:
            u["output_tokens"] = _add(u["output_tokens"], out + (_num(tokens.get("reasoning")) or 0))
        u["cost_usd"] = _add(u["cost_usd"], _num(part.get("cost")))
    if steps:
        u["turns"] = steps
    return u


_IN = ("input_tokens", "prompt_tokens", "inputTokens", "promptTokens")
_OUT = ("output_tokens", "completion_tokens", "outputTokens", "completionTokens")
_CACHED = ("cached_tokens", "cached_input_tokens", "cache_read_input_tokens", "cacheReadTokens")
_COST = ("total_cost_usd", "cost_usd", "costUSD", "cost")


def _walk(value):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def parse_generic(values):
    """Unknown format: the last object that looks like a usage block, the last cost field.
    Totals are taken as printed; a CLI that prints per-step usage needs its own parser."""
    u = empty_usage()
    for d in _walk(values):
        found_in = next((_num(d[k]) for k in _IN if k in d and _num(d[k]) is not None), None)
        found_out = next((_num(d[k]) for k in _OUT if k in d and _num(d[k]) is not None), None)
        if found_in is not None or found_out is not None:
            u["input_tokens"], u["output_tokens"] = found_in, found_out
            u["cached_tokens"] = next((_num(d[k]) for k in _CACHED if k in d), None)
        cost = next((_num(d[k]) for k in _COST if k in d and _num(d[k]) is not None), None)
        if cost is not None:
            u["cost_usd"] = cost
    return u


PARSERS = {"claude": parse_claude, "codex": parse_codex, "opencode": parse_opencode,
           "generic": parse_generic, "none": lambda values: empty_usage()}


def parse_usage(parser, stdout_text):
    if parser not in PARSERS:
        raise ValueError(f"unknown parser {parser!r}; known: {', '.join(PARSERS)}")
    return PARSERS[parser](iter_json(stdout_text))


def estimate_cost(usage, price):
    """Cost from tokens and an agents.json `price` block (USD per 1M tokens), same formula as bench/cost.py."""
    if not price or usage["input_tokens"] is None or usage["output_tokens"] is None:
        return None
    import sys
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from bench.cost import request_cost
    cached = min(usage["cached_tokens"] or 0, usage["input_tokens"])
    return request_cost(usage["input_tokens"], usage["output_tokens"], cached, price)
