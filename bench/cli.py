"""Command line: list | prompts | run | score | report. See README.md."""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from . import TASKSET_VERSION, __version__
from .checks import check
from .client import APIError, chat
from .cost import request_cost
from .tasks import load_tasks

ROOT = Path(__file__).resolve().parent.parent


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _record(task, run, label, model, text, result, usage=None, latency=None, cost=None, source="api"):
    return {
        "taskset": TASKSET_VERSION, "bench_version": __version__,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": label, "model": model, "source": source,
        "task_id": task.id, "task_version": task.version, "category": task.category, "run": run,
        "passed": result.passed, "details": result.summary(),
        "rules": [r.__dict__ for r in result.rules],
        "usage": usage, "latency_s": latency, "cost_usd": cost, "response": text,
    }


def cmd_list(args):
    for t in load_tasks(args.tasks):
        print(f"{t.id:18} {t.category:8} {t.language}  {t.title}")


def cmd_prompts(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in load_tasks(args.tasks):
        (out / f"{t.id}.txt").write_text(t.prompt + "\n", encoding="utf-8")
    print(f"prompts written to {out}/ — paste each into the chat, save answers as <task_id>.txt, then run `score`")


def _price(args):
    if args.price_input is None:
        return None
    price = {"input": args.price_input, "output": args.price_output,
             "cached": args.price_cached if args.price_cached is not None else args.price_input}
    if args.long_threshold:
        price.update(long_threshold=args.long_threshold, long_input_mult=args.long_input_mult,
                     long_output_mult=args.long_output_mult)
    return price


def cmd_run(args):
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        print(f"warning: environment variable {args.api_key_env} is empty; sending requests without a key",
              file=sys.stderr)
    temperature = None if str(args.temperature).lower() == "none" else float(args.temperature)
    price = _price(args)
    label = args.label or args.model
    tasks = load_tasks(args.tasks)
    out = ROOT / "results" / _safe(label) / f"{_now()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    passed = total = 0
    with out.open("w", encoding="utf-8") as f:
        for run in range(1, args.runs + 1):
            for t in tasks:
                try:
                    r = chat(args.base_url, api_key, args.model, t.prompt, temperature=temperature,
                             max_tokens=args.max_tokens, timeout=args.timeout)
                except APIError as e:
                    print(f"[run {run}] {t.id:18} API ERROR {e}", file=sys.stderr)
                    continue
                result = check(t, r["text"])
                u = r["usage"]
                cost = request_cost(u["input_tokens"], u["output_tokens"],
                                    min(u["cached_tokens"], u["input_tokens"]), price) if price else None
                f.write(json.dumps(_record(t, run, label, r["raw_model"], r["text"], result,
                                           u, r["latency_s"], cost), ensure_ascii=False) + "\n")
                f.flush()
                total += 1
                passed += result.passed
                print(f"[run {run}] {t.id:18} {'PASS' if result.passed else 'FAIL'}  "
                      f"{r['latency_s']:6.1f}s  {u['output_tokens']:6} out tok"
                      + (f"  ${cost:.4f}" if cost is not None else "")
                      + ("" if result.passed else f"  — {result.summary()[:160]}"))
    print(f"\n{passed}/{total} passed. Results: {out.relative_to(ROOT)}")


def cmd_score(args):
    src = Path(args.responses)
    label = args.label or src.name
    tasks = {t.id: t for t in load_tasks(args.tasks)}
    out = ROOT / "results" / _safe(label) / f"{_now()}-manual.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for path in sorted(src.glob("*.*")):
            m = re.match(r"^(?P<id>[a-z0-9_]+?)(?:__run(?P<run>\d+))?\.(txt|md)$", path.name)
            if not m or m.group("id") not in tasks:
                continue
            t = tasks[m.group("id")]
            text = path.read_text(encoding="utf-8")
            result = check(t, text)
            f.write(json.dumps(_record(t, int(m.group("run") or 1), label, args.model or label, text,
                                       result, source="manual"), ensure_ascii=False) + "\n")
            n += 1
            print(f"{path.name:28} {'PASS' if result.passed else 'FAIL'}"
                  + ("" if result.passed else f"  — {result.summary()[:160]}"))
    missing = sorted(set(tasks) - {re.sub(r"__run\d+$", "", p.stem) for p in src.glob('*.*')})
    if missing:
        print("no answers for: " + ", ".join(missing))
    print(f"\nscored {n} answers. Results: {out.relative_to(ROOT)}")


def _load_records(paths):
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [Path(x) for x in glob.glob(str(p))]
    for fp in files:
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def cmd_report(args):
    records = list(_load_records(args.paths or [ROOT / "results"]))
    if not records:
        raise SystemExit("no results found")
    tasks = [t.id for t in load_tasks()]
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    lines = [f"Neurohive base task set {TASKSET_VERSION} — {len(tasks)} tasks", "",
             "| Model | Tasks solved (majority of runs) | Runs passed | Avg latency, s | Total cost, $ | Cost per passed run, $ |",
             "| --- | --- | --- | --- | --- | --- |"]
    matrix = {}
    for label, rs in sorted(by_label.items()):
        per_task = defaultdict(list)
        for r in rs:
            per_task[r["task_id"]].append(r["passed"])
        solved = sum(1 for v in per_task.values() if sum(v) * 2 > len(v))
        runs_passed = sum(r["passed"] for r in rs)
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        total_cost = sum(costs) if costs else None
        per_pass = (total_cost / runs_passed) if (total_cost is not None and runs_passed) else None
        avg_lat = f"{sum(lat) / len(lat):.1f}" if lat else "—"
        cost_s = f"{total_cost:.4f}" if total_cost is not None else "—"
        per_pass_s = f"{per_pass:.4f}" if per_pass is not None else "—"
        lines.append(f"| {label} | {solved}/{len(tasks)} | {runs_passed}/{len(rs)} | "
                     f"{avg_lat} | {cost_s} | {per_pass_s} |")
        matrix[label] = {k: f"{sum(v)}/{len(v)}" for k, v in per_task.items()}

    labels = sorted(matrix)
    lines += ["", "| Task | " + " | ".join(labels) + " |", "| --- |" + " --- |" * len(labels)]
    for tid in tasks:
        lines.append(f"| {tid} | " + " | ".join(matrix[lb].get(tid, "—") for lb in labels) + " |")
    report = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"report written to {args.out}")
    print(report)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench", description="Neurohive base task set")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list tasks")
    s.add_argument("--tasks", nargs="*", help="task id patterns, e.g. code_* ru_qa")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("prompts", help="export prompts for manual testing in web chats")
    s.add_argument("--out", default="prompts")
    s.add_argument("--tasks", nargs="*")
    s.set_defaults(fn=cmd_prompts)

    s = sub.add_parser("run", help="run tasks against an OpenAI-compatible API")
    s.add_argument("--base-url", required=True, help="e.g. https://openrouter.ai/api/v1 or http://localhost:11434/v1")
    s.add_argument("--model", required=True)
    s.add_argument("--label", help="name in reports (default: model)")
    s.add_argument("--api-key-env", default="NH_API_KEY", help="env var with the API key ('' for none)")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--temperature", default="0", help="number or 'none' for models that reject it")
    s.add_argument("--max-tokens", type=int)
    s.add_argument("--timeout", type=int, default=600)
    s.add_argument("--tasks", nargs="*")
    s.add_argument("--price-input", type=float, help="USD per 1M uncached input tokens")
    s.add_argument("--price-output", type=float, help="USD per 1M output tokens")
    s.add_argument("--price-cached", type=float, help="USD per 1M cached input tokens")
    s.add_argument("--long-threshold", type=int)
    s.add_argument("--long-input-mult", type=float, default=1.0)
    s.add_argument("--long-output-mult", type=float, default=1.0)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="score answers saved by hand from web chats")
    s.add_argument("responses", help="folder with <task_id>.txt or <task_id>__run2.txt files")
    s.add_argument("--label")
    s.add_argument("--model", help="exact model/version shown in the chat, for the record")
    s.add_argument("--tasks", nargs="*")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("report", help="summarize results/*.jsonl as Markdown")
    s.add_argument("paths", nargs="*")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    if getattr(args, "price_input", None) is not None and args.price_output is None:
        p.error("--price-output is required with --price-input")
    args.fn(args)


if __name__ == "__main__":
    main()
