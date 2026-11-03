"""One run = clean copy -> setup -> agent -> diff -> hidden tests -> test command -> JSON line."""
import json
import os
import platform
import shutil
import traceback
from datetime import datetime, timezone

from . import agents as ag
from .workspace import commit_all, install_hidden_tests, prepare, snapshot_diff

SCHEMA = 1


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read(path, limit=None):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text if limit is None else text[-limit:]


def _rel(path, start):
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def run_one(task, agent, run_no, *, work_dir, results_dir, out_path, model=None,
            keep_workdir=False, version=None):
    """Run `agent` on `task` once and append the record to `out_path`. Returns the record."""
    run_id = f"{utc_stamp()}-{task.id}-{agent.id}-r{run_no}"
    log_dir = os.path.join(results_dir, "logs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    dest = os.path.join(work_dir, "runs", run_id)
    cache_dir = os.path.join(work_dir, "cache")
    model = model or agent.model
    rec = {
        "schema": SCHEMA, "run_id": run_id, "task_id": task.id, "task_type": task.type,
        "repo": task.repo, "commit": task.commit, "base_commit": None,
        "agent": agent.id, "agent_title": agent.title, "agent_version": version, "model": model,
        "run": run_no, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "agent_exit_code": None, "agent_timed_out": False, "duration_s": None,
        "usage": ag.empty_usage(), "cost_usd": None, "cost_source": None,
        "diff_stats": None, "hidden_tests": [],
        "tests": None, "passed": False, "error": None,
        "logs": {},
    }

    def log(name):
        path = os.path.join(log_dir, name)
        rec["logs"][name.rsplit(".", 1)[0]] = _rel(path, results_dir)
        return path

    try:
        rec["base_commit"] = prepare(task, dest, cache_dir)
        if task.setup_cmd:
            setup = ag.run_shell(task.setup_cmd, dest, log("setup.log"), task.test_timeout_s)
            if setup.exit_code != 0:
                raise RuntimeError(f"setup_cmd failed (exit {setup.exit_code}, timed_out={setup.timed_out})")
            # whatever setup created is part of the base, not the agent's work
            rec["base_commit"] = commit_all(dest, "bench setup")

        cmd = ag.build_command(agent, task.prompt, model)
        with open(log("prompt.md"), "w", encoding="utf-8") as f:
            f.write(task.prompt)
        proc = ag.run_process(cmd, dest, log("agent.stdout.log"), log("agent.stderr.log"),
                              task.agent_timeout_s, env=agent.env,
                              stdin_text=task.prompt if agent.stdin else None)
        rec["agent_exit_code"] = proc.exit_code
        rec["agent_timed_out"] = proc.timed_out
        rec["duration_s"] = proc.duration_s

        usage = ag.parse_usage(agent.parser, _read(os.path.join(log_dir, "agent.stdout.log")))
        rec["usage"] = usage
        if usage["cost_usd"] is not None:
            rec["cost_usd"], rec["cost_source"] = usage["cost_usd"], "agent"
        else:
            est = ag.estimate_cost(usage, agent.price)
            if est is not None:
                rec["cost_usd"], rec["cost_source"] = est, "estimate"

        diff, stats = snapshot_diff(dest, rec["base_commit"])
        with open(log("changes.diff"), "w", encoding="utf-8") as f:
            f.write(diff)
        rec["diff_stats"] = stats

        rec["hidden_tests"] = install_hidden_tests(task, dest, cache_dir)
        t = ag.run_shell(task.test_cmd, dest, log("tests.log"), task.test_timeout_s)
        rec["tests"] = {"cmd": task.test_cmd, "exit_code": t.exit_code, "timed_out": t.timed_out,
                        "duration_s": t.duration_s,
                        "tail": _read(os.path.join(log_dir, "tests.log"), 2000)}
        rec["passed"] = t.exit_code == 0 and not t.timed_out
    except Exception as e:  # a broken run is still a run: it costs money and counts as failed
        rec["error"] = f"{type(e).__name__}: {e}"
        with open(log("error.log"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
    finally:
        if not keep_workdir:
            shutil.rmtree(dest, ignore_errors=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def run_matrix(tasks, agent_list, runs, *, work_dir, results_dir, out_path, models=None,
               keep_workdir=False, progress=print):
    """Run every agent on every task `runs` times. Agents alternate inside each task so that
    API load or time of day does not favour one of them."""
    versions = {a.id: ag.agent_version(a) for a in agent_list}
    records = []
    for run_no in range(1, runs + 1):
        for task in tasks:
            for agent in agent_list:
                progress(f"[run {run_no}/{runs}] {task.id} × {agent.id} ...")
                rec = run_one(task, agent, run_no, work_dir=work_dir, results_dir=results_dir,
                              out_path=out_path, model=(models or {}).get(agent.id),
                              keep_workdir=keep_workdir, version=versions[agent.id])
                cost = "?" if rec["cost_usd"] is None else f"${rec['cost_usd']:.4f}"
                status = "PASS" if rec["passed"] else ("ERROR " + rec["error"] if rec["error"] else "FAIL")
                progress(f"    {status}  {rec['duration_s']}s  {cost}")
                records.append(rec)
    return records
