# Стенд для сравнения агентов программирования

Claude Code, Codex, OpenCode и Kimi Code решают одни и те же задачи в одинаковых условиях. Стенд готовит чистую копию репозитория для каждого прогона, запускает агента без участия человека, прогоняет скрытые тесты и записывает время, токены, стоимость и результат в `results/agents/*.jsonl`. Потом выгружает диффы под буквами A, B, C, D для слепой оценки кода и собирает сводный отчёт с ценой одной решённой задачи.

Нужны Python 3.10+ и git, только стандартная библиотека. Все команды запускаются из корня репозитория.

```bash
python -m experiments.agents.agentbench check                 # какие агенты установлены
python -m experiments.agents.agentbench run --runs 3          # все задачи × все агенты × 3 прогона
python -m experiments.agents.agentbench report --out report.md
python -m experiments.agents.agentbench blind --out blind --seed 2026
```

## Задачи

Описаны в [`tasks.json`](tasks.json). Поля задачи:

| Поле | Что это |
| --- | --- |
| `id`, `type` | тип: `feature` (добавить функцию с тестами), `bugfix` (исправить баг из закрытого issue), `greenfield` (маленькое приложение с нуля) |
| `repo` | открытый git-репозиторий (URL или локальный путь); у `greenfield` — `null`, агент начинает с пустого репозитория |
| `commit` | коммит, с которого стартует агент. У `bugfix` это коммит перед фиксом: можно писать `<sha фикса>^` |
| `prompt` / `prompt_file` | текст задания, которое получает агент |
| `test_cmd` | команда тестов; задача решена, если она завершилась с кодом 0 |
| `hidden_tests` | тесты, которые кладутся в рабочую копию **после** работы агента: `{"from_commit": sha, "path": ...}` берёт файл из коммита репозитория (обычно из коммита с фиксом), `{"file": ..., "path": ...}` — из папки `tasks/` |
| `setup_cmd` | необязательно: подготовка окружения до запуска агента (например, `python -m venv .venv && .venv/bin/pip install -e .`); созданное ею не попадает в дифф агента |
| `agent_timeout_s`, `test_timeout_s` | тайм-ауты, по умолчанию 1800 и 600 с |

Остальные поля (`issue`, `fix_commit`, `source`) записываются для справки.

Сейчас в конфиге три задачи:

- `feature_subfactorial` — добавить `subfactorial(n)` в [more-itertools](https://github.com/more-itertools/more-itertools) (коммит перед [842a2b1](https://github.com/more-itertools/more-itertools/commit/842a2b17a8e3f78462fe20cd4a88abd32085c111)); проверка — тесты из этого коммита;
- `bugfix_bucket_phantom_keys` — баг из закрытого [issue #1284](https://github.com/more-itertools/more-itertools/issues/1284): проверка `in` у `bucket` добавляет «фантомный» ключ; старт с коммита перед фиксом [2b8d5cd](https://github.com/more-itertools/more-itertools/commit/2b8d5cddce4ea01f88a973551cd50479cf5fa45f), проверка — тесты из фикса;
- `greenfield_meter_bills` — консольная программа расчёта платы за воду по показаниям счётчиков; скрытые приёмочные тесты в `tasks/greenfield_meter_bills/hidden_test_meter.py`, эталонное решение для их проверки — `reference_meter.py`.

Фикс и всё, что после него, агент не видит: рабочая копия клонируется только до стартового коммита (`--single-branch`), без remote. Посмотреть ответ через `git log --all` или `git fetch` нельзя.

Задачи для статьи стоит брать свежие, после даты обучения моделей: старые фиксы популярных библиотек модели могли запомнить.

## Агенты

Сами CLI стенд не устанавливает. Команды запуска лежат в [`agents.json`](agents.json); `{prompt}` заменяется текстом задания, `{model}` — моделью из `--model agent=model` или поля `model`. Если CLI нет в PATH, `run` останавливается до первого прогона и печатает, что и как установить (`--skip-missing` запускает остальных).

| Агент | Установка | Неинтерактивный запуск (в `agents.json`) | Токены | Стоимость |
| --- | --- | --- | --- | --- |
| Claude Code | `npm install -g @anthropic-ai/claude-code` | `claude -p "<задание>" --output-format json --dangerously-skip-permissions [--model M]` | да | да (`total_cost_usd`) |
| Codex CLI | `npm install -g @openai/codex` | `codex exec --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "<задание>" [-m M]` | да (`turn.completed`) | нет → оценка по тарифу |
| OpenCode | `npm install -g opencode-ai` | `opencode run --format json "<задание>" [-m provider/model]` | да (`step_finish`) | да |
| Kimi Code | `uv tool install kimi-cli` | `kimi --print --output-format stream-json -c "<задание>" [-m M]` | если выводит | если выводит |

Флаги CLI меняются от версии к версии: перед замером сверьте их с `--help` и поправьте `agents.json`. Версия каждого агента (`version_cmd`) записывается в результаты.

Ключи API передаются через переменные окружения или вход в самом CLI; стенд ничего не хранит. Для агента можно задать `env` (дополнительные переменные) и `stdin: true`, если задание нужно подавать на stdin.

**Токены и стоимость.** Парсер (`parser`) читает stdout агента: `claude`, `codex`, `opencode`, `generic` (ищет блок `usage` и поле `cost` в любом JSON) или `none`. `input_tokens` в результатах — все входные токены, включая кеш; `cached_tokens` — чтение из кеша. Если агент не выводит стоимость, а в `agents.json` у него есть `price` (USD за 1M токенов: `input`, `cached`, `output`, как в `bench/cost.py`), стоимость считается по токенам и помечается `cost_source: "estimate"`. Если нет ни того, ни другого, `cost_usd` будет `null`, а не 0. При работе по подписке, а не по API, стоимость прогона — условная.

**Права.** Агенты запускаются с отключёнными подтверждениями и могут выполнять любые команды. Запускайте стенд в контейнере или виртуальной машине без лишних секретов, с отдельными API-ключами и лимитом трат.

## Прогон

```bash
python -m experiments.agents.agentbench run --tasks bugfix_bucket_phantom_keys --agents claude,codex \
  --runs 3 --model claude=<модель> --model codex=<модель>
```

Для каждого прогона:

1. Чистая копия репозитория на нужном коммите в `--work-dir` (по умолчанию `$TMPDIR/nh-agent-bench`; кеш клонов там же). Папка должна быть **вне** этого репозитория, иначе агент прочитает его `CLAUDE.md`/`AGENTS.md` и ответы. Вручную копию можно сделать командой `prepare <task> <папка>`.
2. `setup_cmd`, если задан.
3. Агент в неинтерактивном режиме с тайм-аутом; по тайм-ауту убивается вся группа процессов.
4. Дифф всех изменений относительно стартового коммита (включая коммиты агента и новые файлы).
5. Скрытые тесты копируются в рабочую копию, запускается `test_cmd`.
6. Строка в `results/agents/<UTC>.jsonl`, логи — в `results/agents/logs/<run_id>/`: `prompt.md`, `agent.stdout.log`, `agent.stderr.log`, `changes.diff`, `tests.log`.

Порядок: прогон 1 всех агентов на всех задачах, потом прогон 2 и так далее; на каждой задаче агенты идут друг за другом, чтобы нагрузка на API и время суток не давали никому преимущества.

Поля записи: `run_id`, `task_id`, `task_type`, `repo`, `commit`, `base_commit`, `agent`, `agent_title`, `agent_version`, `model`, `run`, `started_at`, `duration_s` (время работы агента), `agent_exit_code`, `agent_timed_out`, `usage` (`input_tokens`, `output_tokens`, `cached_tokens`, `cache_write_tokens`, `cost_usd`, `turns`), `cost_usd`, `cost_source` (`agent` / `estimate` / `null`), `diff_stats`, `hidden_tests`, `tests` (`cmd`, `exit_code`, `timed_out`, `duration_s`, `tail`), `passed`, `error`, `logs`.

Упавший прогон (агент вышел с ошибкой, не уложился в тайм-аут, сломалась подготовка) — это неуспешный прогон: деньги потрачены, задача не решена.

## Слепая оценка

```bash
python -m experiments.agents.agentbench blind --out blind --seed 2026 [--run 1] [файлы.jsonl ...]
```

В `blind/<задача>/` лежат `task.md` и `A.diff` … `D.diff`, в `blind/grading.csv` — пустая таблица оценок. Буквы перемешиваются отдельно для каждой задачи (воспроизводимо по `--seed`), поэтому «A» не означает одного и того же агента на всех задачах. Из диффов убираются служебные файлы агентов (`CLAUDE.md`, `AGENTS.md`, `.claude/`, `.codex/`, `opencode.json` и т. п.), а названия агентов, вендоров и моделей заменяются на «[агент]» (`--no-redact` отключает). Берётся один прогон каждого агента (`--run`, по умолчанию первый), чтобы у всех было одинаковое число образцов.

Ключ с соответствием букв и агентов пишется отдельно, в `blind.key.json` рядом с папкой: оценщикам отдаётся только папка.

## Отчёт

```bash
python -m experiments.agents.agentbench report --out report.md [файлы.jsonl ...]
```

По каждому агенту: успешные прогоны, решённые задачи (пройдены в большинстве прогонов), медианное время, токены, траты и **цена решённой задачи = все траты (включая неудачные прогоны) / успешные прогоны**. Ниже — таблица задача × агент и список сбоев. Если у части прогонов стоимость неизвестна, сумма помечается «≥».

## Тесты стенда

```bash
python -m pytest experiments/agents/tests
```

Тесты используют фиктивного агента (`tests/fake_agent.py`) и локальный репозиторий: проверяются конфиг, чистые копии (фикс недоступен), запуск, тайм-аут, ошибка для неустановленного CLI, разбор токенов и стоимости в форматах Claude Code, Codex и OpenCode, слепая выгрузка и расчёт цены решённой задачи.
