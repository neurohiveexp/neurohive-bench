"""Markdown summary. Price of a solved task = all money spent / passed runs."""
import statistics

from .results import load_records


def summarize(records):
    """Per-agent totals. Every run counts toward spend, including failed, timed-out and broken ones."""
    agents = {}
    for r in records:
        a = agents.setdefault(r["agent"], {
            "agent": r["agent"], "title": r.get("agent_title") or r["agent"], "models": set(),
            "runs": 0, "passed": 0, "errors": 0, "timeouts": 0, "durations": [],
            "cost": 0.0, "runs_without_cost": 0, "estimated_runs": 0,
            "input_tokens": 0, "output_tokens": 0, "runs_without_tokens": 0, "per_task": {},
        })
        a["runs"] += 1
        a["passed"] += bool(r["passed"])
        a["errors"] += bool(r.get("error"))
        a["timeouts"] += bool(r.get("agent_timed_out"))
        if r.get("model"):
            a["models"].add(r["model"])
        if r.get("duration_s") is not None:
            a["durations"].append(r["duration_s"])
        if r.get("cost_usd") is None:
            a["runs_without_cost"] += 1
        else:
            a["cost"] += r["cost_usd"]
            a["estimated_runs"] += r.get("cost_source") == "estimate"
        u = r.get("usage") or {}
        if u.get("input_tokens") is None and u.get("output_tokens") is None:
            a["runs_without_tokens"] += 1
        else:
            a["input_tokens"] += u.get("input_tokens") or 0
            a["output_tokens"] += u.get("output_tokens") or 0
        t = a["per_task"].setdefault(r["task_id"], [0, 0])
        t[0] += bool(r["passed"])
        t[1] += 1
    for a in agents.values():
        a["cost_per_solved"] = cost_per_solved(a["cost"], a["passed"]) if a["runs_without_cost"] < a["runs"] else None
        a["solved_tasks"] = sum(1 for p, n in a["per_task"].values() if p * 2 > n)
    return agents


def cost_per_solved(total_cost, passed_runs):
    """All spend divided by passed runs; None when nothing passed (the price is unbounded)."""
    if passed_runs == 0:
        return None
    return total_cost / passed_runs


def _money(x):
    return "—" if x is None else f"${x:.2f}" if x >= 1 else f"${x:.4f}"


def _cost_cell(a):
    if a["runs_without_cost"] == a["runs"]:
        return "нет данных"
    cell = _money(a["cost"])
    if a["runs_without_cost"]:
        cell = "≥ " + cell + f" ({a['runs_without_cost']} без цены)"
    if a["estimated_runs"]:
        cell += f" (оценка: {a['estimated_runs']})"
    return cell


def _per_solved_cell(a):
    if a["runs_without_cost"] == a["runs"]:
        return "нет данных"
    if a["passed"] == 0:
        return "∞ (нет решённых)"
    cell = _money(a["cost_per_solved"])
    return ("≥ " + cell) if a["runs_without_cost"] else cell


def _tokens(n, missing, runs):
    if missing == runs:
        return "—"
    s = f"{n / 1e6:.2f}M" if n >= 100_000 else f"{n:,}".replace(",", " ")
    return s + ("*" if missing else "")


def render(records):
    agents = summarize(records)
    tasks = sorted({r["task_id"] for r in records})
    types = {r["task_id"]: r.get("task_type", "") for r in records}
    lines = ["# Сравнение агентов", ""]
    lines.append(f"Прогонов: {len(records)}, задач: {len(tasks)}, агентов: {len(agents)}.")
    lines.append("")
    lines.append("**Цена решённой задачи** = все траты агента (включая неудачные прогоны) / успешные прогоны. "
                 "Прогон успешен, если команда тестов задачи (со скрытыми тестами) завершилась с кодом 0.")
    lines.append("")
    lines.append("| Агент | Модель | Успешно | Решено задач | Время, медиана | Вход | Выход | Траты | Цена решённой задачи |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for a in sorted(agents.values(), key=lambda a: (-a["passed"] / a["runs"], a["agent"])):
        med = f"{statistics.median(a['durations']):.0f} с" if a["durations"] else "—"
        lines.append(
            f"| {a['title']} | {', '.join(sorted(a['models'])) or '—'} | {a['passed']}/{a['runs']} "
            f"| {a['solved_tasks']}/{len(a['per_task'])} | {med} "
            f"| {_tokens(a['input_tokens'], a['runs_without_tokens'], a['runs'])} "
            f"| {_tokens(a['output_tokens'], a['runs_without_tokens'], a['runs'])} "
            f"| {_cost_cell(a)} | {_per_solved_cell(a)} |")
    lines += ["", "«Решено задач» — задача пройдена в большинстве прогонов. "
              "«≥» — у части прогонов агент не вывел цену, сумма занижена. "
              "«оценка» — цена посчитана по токенам и тарифу из agents.json. "
              "\\* — у части прогонов нет данных о токенах.", ""]
    lines += ["## По задачам", ""]
    order = sorted(agents)
    lines.append("| Задача | Тип | " + " | ".join(agents[a]["title"] for a in order) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in order) + " |")
    for t in tasks:
        cells = []
        for a in order:
            p = agents[a]["per_task"].get(t)
            cells.append("—" if p is None else f"{p[0]}/{p[1]}")
        lines.append(f"| `{t}` | {types[t]} | " + " | ".join(cells) + " |")
    problems = [r for r in records if r.get("error") or r.get("agent_timed_out")]
    if problems:
        lines += ["", "## Сбои", ""]
        for r in problems:
            why = r.get("error") or "агент не уложился в тайм-аут"
            lines.append(f"- `{r['run_id']}`: {why}")
    return "\n".join(lines) + "\n"


def render_paths(paths):
    return render(load_records(paths))
