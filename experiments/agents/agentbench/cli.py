"""python -m experiments.agents.agentbench <command>"""
import argparse
import os
import sys
import tempfile

from . import agents as ag
from .blind import export
from .config import REPO_ROOT, DEFAULT_AGENTS, DEFAULT_RESULTS, DEFAULT_TASKS, ConfigError, load_agents, load_tasks, pick
from .report import render_paths
from .runner import run_matrix, utc_stamp
from .workspace import WorkspaceError, prepare

DEFAULT_WORK = os.path.join(tempfile.gettempdir(), "nh-agent-bench")


def _models(values):
    out = {}
    for v in values or []:
        agent, sep, model = v.partition("=")
        if not sep or not agent or not model:
            raise ConfigError(f"--model expects agent=model, got {v!r}")
        out[agent] = model
    return out


def cmd_check(args):
    agents = pick(load_agents(args.agents_file), args.agents, "agents")
    missing = 0
    for a in agents:
        try:
            path = ag.check_installed(a)
            print(f"OK       {a.id:10} {path}  {ag.agent_version(a) or ''}")
        except ag.AgentNotFound as e:
            missing += 1
            print(f"MISSING  {e}")
    tasks = load_tasks(args.tasks_file)
    print(f"\n{len(tasks)} задач(и) в {args.tasks_file}: " + ", ".join(f"{t.id} ({t.type})" for t in tasks))
    return 2 if missing else 0


def cmd_prepare(args):
    task = pick(load_tasks(args.tasks_file), args.task, "tasks")[0]
    base = prepare(task, args.dest, os.path.join(args.work_dir, "cache"))
    print(f"{task.id}: чистая копия в {args.dest}, базовый коммит {base}")
    return 0


def cmd_run(args):
    tasks = pick(load_tasks(args.tasks_file), args.tasks, "tasks")
    agents = pick(load_agents(args.agents_file), args.agents, "agents")
    models = _models(args.model)
    unknown = set(models) - {a.id for a in agents}
    if unknown:
        raise ConfigError(f"--model for agents that are not selected: {', '.join(sorted(unknown))}")
    available, errors = [], []
    for a in agents:
        try:
            ag.check_installed(a)
            available.append(a)
        except ag.AgentNotFound as e:
            errors.append(str(e))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        if not args.skip_missing or not available:
            print("\nОстановлено: установите агенты или запустите с --skip-missing.", file=sys.stderr)
            return 2
    work_dir = os.path.abspath(args.work_dir)
    results_dir = os.path.abspath(args.results_dir)
    if work_dir == REPO_ROOT or work_dir.startswith(REPO_ROOT + os.sep):
        print(f"warning: work dir {work_dir} is inside the bench repository; agents may read its "
              f"CLAUDE.md/AGENTS.md and answers. Use a directory outside it.", file=sys.stderr)
    out = args.out or os.path.join(results_dir, f"{utc_stamp()}.jsonl")
    records = run_matrix(tasks, available, args.runs, work_dir=work_dir, results_dir=results_dir,
                         out_path=out, models=models, keep_workdir=args.keep_workdirs)
    passed = sum(r["passed"] for r in records)
    print(f"\n{passed}/{len(records)} прогонов прошли тесты. Результаты: {out}")
    return 0


def cmd_blind(args):
    tasks = load_tasks(args.tasks_file) if os.path.exists(args.tasks_file) else []
    key_path, key = export(args.results or [args.results_dir], args.out, seed=args.seed, run=args.run,
                           key_path=args.key, tasks=tasks, redact=not args.no_redact)
    n = sum(len(v) for v in key["tasks"].values())
    print(f"{n} диффов по {len(key['tasks'])} задачам в {args.out}")
    print(f"Ключ (не показывать оценщикам): {key_path}")
    return 0


def cmd_report(args):
    text = render_paths(args.results or [args.results_dir])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Отчёт: {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m experiments.agents.agentbench",
                                description="Стенд для сравнения агентов программирования")
    p.add_argument("--tasks-file", default=DEFAULT_TASKS)
    p.add_argument("--agents-file", default=DEFAULT_AGENTS)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS)
    p.add_argument("--work-dir", default=DEFAULT_WORK,
                   help="где лежат кеш репозиториев и рабочие копии (вне репозитория стенда)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("check", help="какие агенты установлены, какие задачи в конфиге")
    s.add_argument("--agents", help="через запятую; по умолчанию все")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("prepare", help="чистая копия репозитория задачи на нужном коммите")
    s.add_argument("task")
    s.add_argument("dest")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("run", help="прогнать агентов на задачах")
    s.add_argument("--tasks", help="через запятую; по умолчанию все")
    s.add_argument("--agents", help="через запятую; по умолчанию все из agents.json")
    s.add_argument("--runs", type=int, default=1)
    s.add_argument("--model", action="append", metavar="AGENT=MODEL", help="модель для агента, можно несколько раз")
    s.add_argument("--out", help="файл JSONL; по умолчанию results/agents/<UTC>.jsonl")
    s.add_argument("--skip-missing", action="store_true", help="не останавливаться, если часть агентов не установлена")
    s.add_argument("--keep-workdirs", action="store_true", help="не удалять рабочие копии после прогона")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("blind", help="диффы под буквами A, B, C, D для слепой оценки")
    s.add_argument("results", nargs="*", help="файлы или папки JSONL; по умолчанию results/agents")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--run", type=int, default=1, help="какой по счёту прогон брать (по умолчанию 1)")
    s.add_argument("--key", help="куда положить ключ; по умолчанию <out>.key.json рядом с папкой")
    s.add_argument("--no-redact", action="store_true", help="не вычищать названия агентов из диффов")
    s.set_defaults(func=cmd_blind)

    s = sub.add_parser("report", help="сводный отчёт в Markdown")
    s.add_argument("results", nargs="*", help="файлы или папки JSONL; по умолчанию results/agents")
    s.add_argument("--out")
    s.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "runs", 1) < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except (ConfigError, WorkspaceError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
