# Results

`python -m bench run` and `python -m bench score` write here: `results/<label>/<UTC timestamp>.jsonl`, one JSON line per task run.

Main fields: `taskset`, `label`, `model`, `source` (`api` or `manual`), `task_id`, `task_version`, `run`, `passed`, `details`, `rules`, `usage` (`input_tokens`, `output_tokens`, `cached_tokens`), `latency_s`, `cost_usd`, `response`.

`python -m experiments.agents.agentbench run` writes agent runs to `results/agents/<UTC timestamp>.jsonl` with logs and diffs in `results/agents/logs/<run_id>/`; fields are listed in [experiments/agents/README.md](../experiments/agents/README.md).

Results used in a published article are committed here unchanged, with the article link in the commit message.
