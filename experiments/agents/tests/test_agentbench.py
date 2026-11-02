"""Tests for the agent bench itself, with a fake agent (fake_agent.py) and a local upstream repo."""
import json
import os
import subprocess
import sys

import pytest

from experiments.agents.agentbench import agents as ag
from experiments.agents.agentbench.blind import export, sanitize
from experiments.agents.agentbench.cli import main
from experiments.agents.agentbench.config import (DEFAULT_AGENTS, DEFAULT_TASKS, ConfigError,
                                                  load_agents, load_tasks)
from experiments.agents.agentbench.report import cost_per_solved, render, summarize
from experiments.agents.agentbench.workspace import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_agent.py")

CALC_V1 = "def add(a, b):\n    return a - b\n"
CALC_V2 = "def add(a, b):\n    return a + b\n"
CALC_V3 = CALC_V2 + "\n\ndef square(x):\n    return x * x\n"
TEST_V1 = "import unittest\n\nimport calc\n\n\nclass T(unittest.TestCase):\n    def test_import(self):\n        self.assertTrue(calc)\n"
TEST_V2 = TEST_V1 + "\n    def test_add(self):\n        self.assertEqual(calc.add(2, 3), 5)\n"
TEST_V3 = TEST_V2 + "\n    def test_square(self):\n        self.assertEqual(calc.square(4), 16)\n"
APP_TEST = "import unittest\n\nimport app\n\n\nclass T(unittest.TestCase):\n    def test_hello(self):\n        self.assertEqual(app.hello(), 'hi')\n"


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, files, msg):
    for name, text in files.items():
        with open(os.path.join(repo, name), "w") as f:
            f.write(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q")
    c1 = commit(repo, {"calc.py": CALC_V1, "test_calc.py": TEST_V1}, "initial")
    git(repo, "tag", "v0.1")
    c2 = commit(repo, {"calc.py": CALC_V2, "test_calc.py": TEST_V2}, "fix add (closes #1)")
    c3 = commit(repo, {"calc.py": CALC_V3, "test_calc.py": TEST_V3}, "add square")
    git(repo, "tag", "v0.2")
    return {"path": str(repo), "c1": c1, "c2": c2, "c3": c3}


def solution(tmp_path, name, files):
    d = tmp_path / "solutions" / name
    d.mkdir(parents=True)
    for fname, text in files.items():
        (d / fname).write_text(text)
    return str(d)


@pytest.fixture
def bench(tmp_path, upstream):
    """tasks.json with one task of each type, agents.json with fake agents, separate results/work dirs."""
    (tmp_path / "hidden").mkdir()
    (tmp_path / "hidden" / "test_app.py").write_text(APP_TEST)
    tasks = {"tasks": [
        {"id": "feat", "type": "feature", "repo": upstream["path"], "commit": upstream["c2"],
         "prompt": "Add square(x).", "test_cmd": f"{sys.executable} -m unittest -q test_calc",
         "hidden_tests": [{"from_commit": upstream["c3"], "path": "test_calc.py"}]},
        {"id": "bug", "type": "bugfix", "repo": upstream["path"], "commit": upstream["c2"] + "^",
         "prompt": "add() subtracts, fix it.", "test_cmd": f"{sys.executable} -m unittest -q test_calc",
         "hidden_tests": [{"from_commit": upstream["c2"], "path": "test_calc.py"}],
         "issue": "https://example.org/issues/1"},
        {"id": "green", "type": "greenfield", "repo": None, "commit": None,
         "prompt": "Write app.py with hello() returning 'hi'.",
         "test_cmd": f"{sys.executable} -m unittest -q test_app",
         "hidden_tests": [{"file": "hidden/test_app.py", "path": "test_app.py"}]},
    ]}
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    sol_all = solution(tmp_path, "all", {"calc.py": CALC_V3, "app.py": "def hello():\n    return 'hi'\n"})

    def fake(mode="solve", fmt="claude", **env):
        e = {"FAKE_MODE": mode, "FAKE_FORMAT": fmt, "FAKE_SOLUTION": sol_all}
        e.update({k.upper(): str(v) for k, v in env.items()})
        return {"title": f"Fake {mode}/{fmt}", "cmd": [sys.executable, FAKE, "{prompt}"],
                "model_args": ["--model", "{model}"], "parser": fmt if fmt != "text" else "generic",
                "env": e, "install": "pip install fake-agent"}

    agents = {
        "good": fake("solve", "claude", fake_trace=tmp_path / "trace.json"),
        "codexish": fake("solve", "codex", fake_commit=1),
        "lazy": fake("fail", "opencode"),
        "crashy": fake("crash", "text"),
        "ghost": {"title": "Ghost", "cmd": ["nh-no-such-agent-cli", "{prompt}"], "install": "brew install ghost"},
    }
    agents["codexish"]["price"] = {"input": 1.0, "cached": 0.1, "output": 10.0}
    (tmp_path / "agents.json").write_text(json.dumps(agents, default=str))
    base = ["--tasks-file", str(tmp_path / "tasks.json"), "--agents-file", str(tmp_path / "agents.json"),
            "--results-dir", str(tmp_path / "results" / "agents"), "--work-dir", str(tmp_path / "work")]
    return {"tmp": tmp_path, "args": base, "upstream": upstream,
            "results": tmp_path / "results" / "agents", "out": tmp_path / "results" / "agents" / "run.jsonl"}


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


# ---------------------------------------------------------------- config

def test_shipped_configs_load():
    tasks = load_tasks(DEFAULT_TASKS)
    assert {t.type for t in tasks} == {"feature", "bugfix", "greenfield"}
    for t in tasks:
        assert t.prompt.strip() and t.test_cmd and t.hidden_tests
        if t.type != "greenfield":
            assert t.repo.startswith("https://") and len(t.commit.rstrip("^")) == 40
    bug = next(t for t in tasks if t.type == "bugfix")
    assert bug.meta["issue"] and bug.commit == bug.meta["fix_commit"] + "^"
    agents = load_agents(DEFAULT_AGENTS)
    assert set(agents) == {"claude", "codex", "opencode", "kimi"}
    assert all(a.install for a in agents.values())


def test_greenfield_hidden_test_passes_reference(tmp_path):
    task_dir = os.path.join(HERE, "..", "tasks", "greenfield_meter_bills")
    env = {**os.environ, "METER_APP": os.path.join(task_dir, "reference_meter.py")}
    p = subprocess.run([sys.executable, "-m", "unittest", "-q", "hidden_test_meter"], cwd=task_dir,
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("bad, message", [
    ({"id": "x", "type": "refactor", "prompt": "p", "test_cmd": "t"}, "type must be"),
    ({"id": "x", "type": "bugfix", "prompt": "p", "test_cmd": "t"}, "needs 'repo' and 'commit'"),
    ({"id": "x", "type": "greenfield", "repo": "https://x/y", "prompt": "p", "test_cmd": "t"}, "empty repo"),
    ({"id": "x", "type": "greenfield", "prompt": "p"}, "test_cmd"),
    ({"id": "x", "type": "greenfield", "test_cmd": "t"}, "prompt"),
    ({"id": "x", "type": "greenfield", "prompt": "p", "test_cmd": "t",
      "hidden_tests": [{"from_commit": "abc", "path": "t.py"}]}, "need a repo"),
])
def test_task_validation(tmp_path, bad, message):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [bad]}))
    with pytest.raises(ConfigError, match=message):
        load_tasks(str(path))


def test_agent_needs_prompt_placeholder(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(json.dumps({"x": {"cmd": ["x", "--go"]}}))
    with pytest.raises(ConfigError, match="prompt"):
        load_agents(str(path))


# ---------------------------------------------------------------- workspace

def test_prepare_bugfix_hides_the_fix(bench, tmp_path):
    task = next(t for t in load_tasks(str(tmp_path / "tasks.json")) if t.id == "bug")
    dest = tmp_path / "ws"
    base = prepare(task, str(dest), str(tmp_path / "cache"))
    up = bench["upstream"]
    assert base == up["c1"]
    assert (dest / "calc.py").read_text() == CALC_V1
    # the fix and everything after it are not in the copy, there is no remote to fetch them from
    for sha in (up["c2"], up["c3"]):
        assert subprocess.run(["git", "cat-file", "-e", sha], cwd=dest).returncode != 0
    assert git(dest, "remote") == ""
    assert "v0.2" not in git(dest, "tag")
    # a second prepare wipes whatever the previous run left
    (dest / "junk.txt").write_text("x")
    (dest / "calc.py").write_text("broken")
    prepare(task, str(dest), str(tmp_path / "cache"))
    assert not (dest / "junk.txt").exists() and (dest / "calc.py").read_text() == CALC_V1
    assert git(dest, "status", "--porcelain") == ""


def test_prepare_greenfield_is_empty(bench, tmp_path):
    task = next(t for t in load_tasks(str(tmp_path / "tasks.json")) if t.id == "green")
    dest = tmp_path / "ws"
    base = prepare(task, str(dest), str(tmp_path / "cache"))
    assert sorted(os.listdir(dest)) == [".git"]
    assert git(dest, "rev-parse", "HEAD") == base


def test_unknown_commit_is_a_clear_error(bench, tmp_path, capsys):
    data = json.loads((tmp_path / "tasks.json").read_text())
    data["tasks"][0]["commit"] = "0" * 40
    (tmp_path / "tasks.json").write_text(json.dumps(data))
    assert main(bench["args"] + ["prepare", "feat", str(tmp_path / "ws")]) == 2
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------- running

def test_run_all_task_types(bench):
    rc = main(bench["args"] + ["run", "--agents", "good,codexish,lazy,crashy", "--out", str(bench["out"]),
                               "--model", "good=fake-model-1"])
    assert rc == 0
    recs = read_jsonl(bench["out"])
    assert len(recs) == 3 * 4
    by = {(r["task_id"], r["agent"]): r for r in recs}

    for task in ("feat", "bug", "green"):
        good = by[(task, "good")]
        assert good["passed"] is True, good
        assert good["tests"]["exit_code"] == 0
        assert good["agent_exit_code"] == 0 and good["error"] is None
        assert good["model"] == "fake-model-1"
        assert good["cost_usd"] == 0.25 and good["cost_source"] == "agent"
        assert good["usage"]["input_tokens"] == 10000 and good["usage"]["cached_tokens"] == 8000
        assert good["usage"]["output_tokens"] == 500 and good["usage"]["turns"] == 4
        assert good["duration_s"] > 0
        assert by[(task, "lazy")]["passed"] is False
        assert by[(task, "crashy")]["agent_exit_code"] == 3 and by[(task, "crashy")]["passed"] is False

    # the bugfix base fails the hidden test until the agent fixes it
    assert "test_add" in by[("bug", "lazy")]["tests"]["tail"]
    # codex: summed turns, no cost printed -> estimated from agents.json price
    cx = by[("feat", "codexish")]
    assert cx["passed"] and cx["usage"]["input_tokens"] == 6000 and cx["usage"]["cached_tokens"] == 2000
    assert cx["cost_source"] == "estimate"
    assert cx["cost_usd"] == pytest.approx((4000 * 1.0 + 2000 * 0.1 + 400 * 10.0) / 1e6)
    # opencode: summed step_finish cost, input includes cache reads, output includes reasoning
    lz = by[("feat", "lazy")]
    assert lz["cost_usd"] == pytest.approx(0.02) and lz["usage"]["input_tokens"] == 2000
    assert lz["usage"]["output_tokens"] == 120
    # crashy prints no usage: tokens and cost unknown, not zero
    assert by[("feat", "crashy")]["cost_usd"] is None and by[("feat", "crashy")]["usage"]["input_tokens"] is None


def test_agent_sees_clean_repo_and_prompt(bench):
    main(bench["args"] + ["run", "--agents", "good", "--tasks", "bug", "--out", str(bench["out"])])
    trace = json.loads((bench["tmp"] / "trace.json").read_text())
    assert trace["prompt"].strip() == "add() subtracts, fix it."
    assert trace["log"].split() == ["initial"]  # no fix commit in history
    assert not trace["cwd"].startswith(str(bench["upstream"]["path"]))


def test_diff_and_logs(bench):
    main(bench["args"] + ["run", "--agents", "good,codexish,crashy", "--tasks", "feat", "--out", str(bench["out"])])
    recs = {r["agent"]: r for r in read_jsonl(bench["out"])}
    res = bench["results"]
    for r in recs.values():
        for name in ("prompt", "agent.stdout", "agent.stderr", "changes", "tests"):
            assert (res / r["logs"][name]).is_file(), (r["agent"], name)
    diff = (res / recs["good"]["logs"]["changes"]).read_text()
    assert "+def square(x):" in diff
    assert "test_calc.py" not in diff  # hidden tests are installed after the diff is taken
    # calc.py gains square() (4 lines), app.py is a new 2-line file from the shared fake solution
    assert recs["good"]["diff_stats"] == {"files": 2, "added": 6, "removed": 0}
    # an agent that commits its work still gets a diff against the task commit
    assert "+def square(x):" in (res / recs["codexish"]["logs"]["changes"]).read_text()
    assert "half_done.txt" in (res / recs["crashy"]["logs"]["changes"]).read_text()
    assert "boom" in (res / recs["crashy"]["logs"]["agent.stderr"]).read_text()
    # work dirs are removed unless asked to keep them
    assert os.listdir(bench["tmp"] / "work" / "runs") == []


def test_timeout_kills_agent(bench, tmp_path):
    data = json.loads((tmp_path / "tasks.json").read_text())
    data["tasks"][2]["agent_timeout_s"] = 1
    (tmp_path / "tasks.json").write_text(json.dumps(data))
    agents = json.loads((tmp_path / "agents.json").read_text())
    agents["sleepy"] = dict(agents["lazy"], env={**agents["lazy"]["env"], "FAKE_MODE": "sleep", "FAKE_SLEEP": "60"})
    (tmp_path / "agents.json").write_text(json.dumps(agents))
    main(bench["args"] + ["run", "--agents", "sleepy", "--tasks", "green", "--out", str(bench["out"])])
    (rec,) = read_jsonl(bench["out"])
    assert rec["agent_timed_out"] is True and rec["agent_exit_code"] is None
    assert rec["duration_s"] < 30 and rec["passed"] is False


def test_setup_cmd_is_not_part_of_the_diff(bench, tmp_path):
    data = json.loads((tmp_path / "tasks.json").read_text())
    data["tasks"][0]["setup_cmd"] = "echo generated > setup_artifact.txt"
    (tmp_path / "tasks.json").write_text(json.dumps(data))
    main(bench["args"] + ["run", "--agents", "good", "--tasks", "feat", "--out", str(bench["out"])])
    (rec,) = read_jsonl(bench["out"])
    assert rec["passed"]
    assert "setup_artifact" not in (bench["results"] / rec["logs"]["changes"]).read_text()


def test_failed_setup_is_recorded_as_failed_run(bench, tmp_path):
    data = json.loads((tmp_path / "tasks.json").read_text())
    data["tasks"][0]["setup_cmd"] = "exit 7"
    (tmp_path / "tasks.json").write_text(json.dumps(data))
    main(bench["args"] + ["run", "--agents", "good", "--tasks", "feat", "--out", str(bench["out"])])
    (rec,) = read_jsonl(bench["out"])
    assert rec["passed"] is False and "setup_cmd failed" in rec["error"]


def test_missing_agent_is_a_clear_error(bench, capsys):
    rc = main(bench["args"] + ["run", "--agents", "good,ghost", "--out", str(bench["out"])])
    err = capsys.readouterr().err
    assert rc == 2
    assert "nh-no-such-agent-cli" in err and "brew install ghost" in err and "--skip-missing" in err
    assert not bench["out"].exists()  # nothing ran


def test_skip_missing_runs_the_rest(bench, capsys):
    rc = main(bench["args"] + ["run", "--agents", "good,ghost", "--tasks", "green", "--skip-missing",
                               "--out", str(bench["out"])])
    assert rc == 0
    assert [r["agent"] for r in read_jsonl(bench["out"])] == ["good"]
    assert "nh-no-such-agent-cli" in capsys.readouterr().err


def test_check_command(bench, capsys):
    assert main(bench["args"] + ["check", "--agents", "good"]) == 0
    assert main(bench["args"] + ["check"]) == 2
    out = capsys.readouterr().out
    assert "MISSING" in out and "ghost" in out


def test_default_results_file_and_runs(bench):
    assert main(bench["args"] + ["run", "--agents", "good", "--tasks", "green", "--runs", "2"]) == 0
    files = [f for f in os.listdir(bench["results"]) if f.endswith(".jsonl")]
    assert len(files) == 1
    recs = read_jsonl(bench["results"] / files[0])
    assert [r["run"] for r in recs] == [1, 2]
    assert len({r["run_id"] for r in recs}) == 2


# ---------------------------------------------------------------- parsers

def test_parsers_on_real_shapes():
    claude = json.dumps({"type": "result", "total_cost_usd": 0.5, "num_turns": 3,
                         "usage": {"input_tokens": 10, "cache_read_input_tokens": 80,
                                   "cache_creation_input_tokens": 10, "output_tokens": 7}})
    u = ag.parse_usage("claude", claude)
    assert (u["input_tokens"], u["cached_tokens"], u["output_tokens"], u["cost_usd"]) == (100, 80, 7, 0.5)
    # stream-json: totals come from the last result line only
    stream = "\n".join([json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 999}}}), claude])
    assert ag.parse_usage("claude", stream)["input_tokens"] == 100
    old_codex = json.dumps({"msg": {"type": "token_count", "info": {"total_token_usage": {
        "input_tokens": 50, "cached_input_tokens": 5, "output_tokens": 6}}}})
    assert ag.parse_usage("codex", "noise\n" + old_codex)["input_tokens"] == 50
    generic = "log line\n" + json.dumps({"usage": {"prompt_tokens": 12, "completion_tokens": 3}, "cost": 0.01})
    u = ag.parse_usage("generic", generic)
    assert (u["input_tokens"], u["output_tokens"], u["cost_usd"]) == (12, 3, 0.01)
    assert ag.parse_usage("generic", "plain text") == ag.empty_usage()
    with pytest.raises(ValueError):
        ag.parse_usage("nope", "")


def test_build_command_substitutes_prompt_and_model():
    agents = load_agents(DEFAULT_AGENTS)
    cmd = ag.build_command(agents["codex"], "do it", "gpt-x")
    assert cmd[-3:] == ["do it", "-m", "gpt-x"] and cmd[:2] == ["codex", "exec"]
    assert "{model}" not in " ".join(ag.build_command(agents["claude"], "p"))


# ---------------------------------------------------------------- blind export

def test_blind_export(bench, tmp_path):
    agents = json.loads((tmp_path / "agents.json").read_text())
    agents["leaky"] = dict(agents["good"], env={**agents["good"]["env"], "FAKE_MODE": "leak"})
    (tmp_path / "agents.json").write_text(json.dumps(agents))
    main(bench["args"] + ["run", "--agents", "good,codexish,lazy,leaky", "--runs", "2",
                          "--out", str(bench["out"])])
    out = tmp_path / "blind"
    assert main(bench["args"] + ["blind", str(bench["out"]), "--out", str(out), "--seed", "7"]) == 0
    key = json.loads((tmp_path / "blind.key.json").read_text())
    assert not (out / "key.json").exists()
    assert set(key["tasks"]) == {"feat", "bug", "green"}
    orders = []
    for task, letters in key["tasks"].items():
        assert sorted(letters) == ["A", "B", "C", "D"]
        assert sorted(v["agent"] for v in letters.values()) == ["codexish", "good", "lazy", "leaky"]
        assert all(v["run_id"].endswith("-r1") for v in letters.values())
        assert sorted(os.listdir(out / task)) == ["A.diff", "B.diff", "C.diff", "D.diff", "task.md"]
        orders.append(tuple(letters[k]["agent"] for k in "ABCD"))
        for letter in "ABCD":
            text = (out / task / f"{letter}.diff").read_text().lower()
            for name in ("claude", "anthropic", "codex", "codexish", "leaky", "good", "fake"):
                assert name not in text, (task, letter, name)
        leaky = next(k for k, v in letters.items() if v["agent"] == "leaky")
        assert key["tasks"][task][leaky]["dropped_files"] == ["CLAUDE.md"]
        assert "[агент]" in (out / task / f"{leaky}.diff").read_text()
    assert len(set(orders)) > 1  # letters are shuffled per task
    rows = (out / "grading.csv").read_text().splitlines()
    assert rows[0].startswith("task_id,letter") and len(rows) == 1 + 3 * 4
    # same seed -> same letters; different seed -> a different assignment somewhere
    main(bench["args"] + ["blind", str(bench["out"]), "--out", str(tmp_path / "b2"), "--seed", "7"])
    assert json.loads((tmp_path / "b2.key.json").read_text())["tasks"] == key["tasks"]


def test_sanitize_keeps_code():
    diff = ("diff --git a/AGENTS.md b/AGENTS.md\n+notes\n"
            "diff --git a/x.py b/x.py\n+total = opusculum + 1  # by Kimi\n+import openai_compat\n")
    text, dropped, n = sanitize(diff)
    assert dropped == ["AGENTS.md"] and "AGENTS" not in text
    assert "opusculum" in text and "openai_compat" in text and "Kimi" not in text and n == 1


# ---------------------------------------------------------------- report

def rec(agent, task, passed, cost, run=1, **kw):
    return {"agent": agent, "agent_title": agent.title(), "task_id": task, "task_type": "feature",
            "run": run, "run_id": f"{task}-{agent}-{run}", "passed": passed, "cost_usd": cost,
            "cost_source": "agent" if cost is not None else None, "duration_s": 10.0,
            "usage": {"input_tokens": 1000, "output_tokens": 100}, **kw}


def test_cost_per_solved_counts_failed_runs():
    assert cost_per_solved(3.0, 2) == 1.5
    assert cost_per_solved(3.0, 0) is None
    records = [rec("a", "t1", True, 1.0), rec("a", "t1", False, 2.0, run=2), rec("a", "t2", True, 0.5),
               rec("b", "t1", False, 1.0), rec("b", "t2", False, 1.0)]
    s = summarize(records)
    assert s["a"]["cost"] == 3.5 and s["a"]["passed"] == 2 and s["a"]["cost_per_solved"] == 1.75
    assert s["a"]["solved_tasks"] == 1  # t1 passed 1 of 2 runs: not a majority
    assert s["b"]["cost_per_solved"] is None
    md = render(records)
    assert "$1.75" in md and "∞ (нет решённых)" in md
    assert "| `t1` | feature | 1/2 | 0/1 |" in md


def test_report_marks_missing_cost():
    records = [rec("a", "t1", True, None), rec("a", "t2", True, None),
               rec("b", "t1", True, 1.0), rec("b", "t2", False, None, error="RuntimeError: x")]
    md = render(records)
    a_line = next(l for l in md.splitlines() if l.startswith("| A |"))
    b_line = next(l for l in md.splitlines() if l.startswith("| B |"))
    assert "нет данных" in a_line
    assert b_line.endswith("| ≥ $1.00 (1 без цены) | ≥ $1.00 |")
    assert "## Сбои" in md and "RuntimeError: x" in md


def test_report_command(bench):
    main(bench["args"] + ["run", "--agents", "good,lazy", "--out", str(bench["out"])])
    out = bench["tmp"] / "report.md"
    assert main(bench["args"] + ["report", "--out", str(out)]) == 0
    md = out.read_text()
    good = next(l for l in md.splitlines() if l.startswith("| Fake solve/claude |"))
    assert "3/3" in good and "$0.7500" in good and "$0.2500" in good
    lazy = next(l for l in md.splitlines() if l.startswith("| Fake fail/opencode |"))
    assert "0/3" in lazy and "∞" in lazy
