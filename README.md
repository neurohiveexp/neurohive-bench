# Neurohive base task set (v1)

12 small, fully checkable tasks that [neurohive.io](https://neurohive.io) uses in its own model tests: local models vs cloud, API cost per solved task, coding agents, blind tests in Russian. One shared set makes results comparable across articles and lets anyone rerun our numbers.

[Русская версия](README.ru.md)

## Tasks

| ID | Category | Lang | What it tests |
| --- | --- | --- | --- |
| `ru_summary` | text_ru | ru | Summarize a news item in exactly 3 bullets, ≤ 60 words, keep key numbers |
| `ru_letter` | text_ru | ru | Formal complaint letter that keeps all 10 given facts, ≤ 220 words |
| `ru_edit` | text_ru | ru | Fix 9 spelling, grammar and punctuation errors without rewriting |
| `ru_qa` | text_ru | ru | Answer from a tariff text: one calculation with a trap, one honest "нет данных" |
| `en_summary` | text_en | en | One-sentence summary, ≤ 35 words, keep key numbers |
| `en_translate` | text_en | en | Russian → English with a mandatory glossary |
| `code_parse_date` | code | ru | Python parser for Russian dates; hidden unit tests incl. invalid dates |
| `code_bugfix` | code | ru | Fix 5 bugs in two pagination functions |
| `code_cost` | code | ru | API request cost with cache and long-context surcharge |
| `json_receipts` | json | ru | Four OCR'd receipts → JSON array (traps: change vs total, legal entity vs store, dd/mm/yy) |
| `json_model_card` | json | en | Press release → model card JSON with distractor numbers |
| `logic_pricing` | logic | ru | Subscription vs pay-as-you-go: break-even point and yearly savings |

All texts are original and fictional (city of Verkhnedolsk, Port of Halvik, Larkspur Labs), so models cannot have memorized the answers.

## How checking works

Every task has deterministic rules in `tasks/<id>/task.json`; a task passes only if all its rules pass. No LLM judge.

- **Text:** word and bullet limits, required facts (regex groups), script checks, required fixes, similarity to the reference.
- **Code:** the answer's Python code runs against hidden unit tests in `tasks/<id>/tests.py`.
- **JSON:** parsed and compared field by field with `reference.json`.
- **Logic and Q&A:** numbers in a fixed answer format.

Rule-based checks confirm facts, format and constraints, not style. For style, add human ratings (as in our blind test).

## Quick start

Python 3.10+, standard library only.

```bash
# model IDs and prices below are examples: take exact values from the provider on test day
python -m bench list

# any OpenAI-compatible API: OpenAI, OpenRouter, DeepSeek, vLLM, Ollama, llama.cpp, LM Studio
export NH_API_KEY=...
python -m bench run --base-url https://openrouter.ai/api/v1 --model deepseek/deepseek-v4.1-flash \
  --runs 3 --price-input 0.15 --price-output 0.60 --price-cached 0.03

# local model, no key
python -m bench run --base-url http://localhost:11434/v1 --model qwen3.8:27b --api-key-env ""

python -m bench report --out report.md
```

Prices are USD per 1M tokens; check the provider's page on the day of the test. Reasoning models that reject `temperature`: pass `--temperature none`.

**Web chats without an API:**

1. Run `python -m bench prompts --out prompts` and paste each prompt into the chat.
2. Save each answer as `responses/<label>/<task_id>.txt`.
3. Run `python -m bench score responses/<label> --model "<version shown in chat>"`.

## Results and metrics

- `results/<label>/<timestamp>.jsonl` stores one line per task run: answer, verdict, rule details, tokens, latency, cost.
- **Solved:** the task passed in the majority of runs.
- **Cost per passed run:** all money spent, including failed runs, divided by passed runs. This is the "price of a solved task" used in our articles.

## Tools

[`tools/cost-calculator`](tools/cost-calculator/) is a single-page calculator (in Russian) of monthly API cost for three typical scenarios, with break-even points against $20 / $200 subscriptions and an own RTX 4090. It uses the same cost formula as `bench/cost.py`; test with `node --test tools/cost-calculator/test.mjs`.

## Experiments

[`experiments/agents`](experiments/agents/) compares coding agents (Claude Code, Codex, OpenCode, Kimi Code) on the same repository tasks: a clean checkout per run, non-interactive agent runs, hidden tests, time/tokens/cost in `results/agents/*.jsonl`, blind A/B/C/D diff export and cost per solved task. Docs in Russian; tests: `python -m pytest experiments/agents/tests`.

## Safety

Code tasks execute model-generated code in a subprocess with a timeout. Run the bench in a container or VM, not on a machine with secrets. Use a separate API key with a spending limit.

## Versioning

Published tasks are never edited. A fix or a new task bumps the task `version` and the set version (`v2`) and is listed in [CHANGELOG.md](CHANGELOG.md). Articles state which version they used.

## License

- Code (`bench/`, `tests/`, `tasks/*/tests.py`, `tasks/*/reference.py`): [MIT](LICENSE).
- Task texts, prompts and reference answers: [CC BY 4.0](TASKS_LICENSE.md), credit "Neurohive base task set, neurohive.io".
