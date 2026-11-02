"""Every reference answer must pass its task; typical wrong answers must fail."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bench.checks import check
from bench.cli import main
from bench.cost import request_cost
from bench.tasks import load_tasks

TASKS = {t.id: t for t in load_tasks()}


def test_twelve_tasks():
    assert len(TASKS) == 12
    cats = [t.category for t in TASKS.values()]
    assert cats.count("text_ru") == 4 and cats.count("text_en") == 2
    assert cats.count("code") == 3 and cats.count("json") == 2 and cats.count("logic") == 1


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_reference_passes(task_id):
    task = TASKS[task_id]
    result = check(task, task.reference())
    assert result.passed, result.summary()


def test_reference_wrapped_in_fences_and_chatter():
    t = TASKS["code_cost"]
    resp = "Вот решение:\n\n```python\n" + t.reference() + "\n```\n\nНадеюсь, помог!"
    assert check(t, resp).passed
    t = TASKS["json_model_card"]
    resp = "Here is the JSON:\n```json\n" + t.reference() + "\n```"
    assert check(t, resp).passed


WRONG = {
    "ru_summary": [
        # 4 bullets
        "- С 1 апреля 2027 года вводят правила.\n- Скорость в центре 15 км/ч.\n- 120 зон парковки.\n- Ночью с 23:00 аренды нет.",
        # no numbers
        "- Город вводит правила для самокатов.\n- В центре скорость ограничат.\n- Парковка будет только в специальных местах.",
    ],
    "ru_letter": [
        "Прошу устранить протечку крыши в квартире 37 дома 14 по ул. Садовой. Иванова М. П.",
    ],
    "ru_edit": [
        # one error left (ошибаеться)
        "Вчера в нашем офисе прошла встреча, посвящённая внедрению нейросетей. Руководитель отдела рассказал, что модель помогает отвечать клиентам вдвое быстрее. Тем не менее сотрудники отметили, что ответы нужно проверять: иногда модель ошибаеться в датах и ценах. В итоге решили, что пока нейросеть будет только предлагать черновики, а отправлять их будут люди.",
        # rewritten instead of corrected
        "На встрече обсудили нейросети: они ускоряют ответы клиентам вдвое, но ошибаются в датах и ценах, поэтому пока нейросеть только предлагает черновики, а отправляют их люди.",
    ],
    "ru_qa": [
        "1) 1125,6 рубля\n2) нет данных",                       # applied the electric-stove coefficient
        "1) 1608 рублей\n2) Для садовых товариществ — одноставочный тариф.",  # hallucination
    ],
    "en_summary": [
        "The Port of Halvik is automating. It will spend €220 million on 14 cranes by 2028.",  # two sentences
    ],
    "en_translate": [
        "A model with open weights can be run on your own server. It needs a video card with 24 GB of memory and quantization to 4 bits. Speed was 38 tokens per second and the context window was 128 thousand tokens.",  # GPU not used
    ],
    "code_parse_date": [
        "```python\nimport datetime\n\ndef parse_ru_date(s):\n    d, m, y = s.split('.')\n    return datetime.date(int(y), int(m), int(d))\n```",
    ],
    "code_bugfix": [
        # the original buggy code
        "```python\ndef paginate(items, page, per_page=10):\n    start = page * per_page\n    end = start + per_page - 1\n    return items[start:end]\n\n\ndef total_pages(n_items, per_page=10):\n    return n_items // per_page\n```",
        # fixed slices but no validation
        "```python\ndef paginate(items, page, per_page=10):\n    start = (page - 1) * per_page\n    return items[start:start + per_page]\n\n\ndef total_pages(n_items, per_page=10):\n    return -(-n_items // per_page)\n```",
    ],
    "code_cost": [
        # multiplies cached price too
        "```python\ndef request_cost(i, o, c, p):\n    if min(i, o, c) < 0 or c > i:\n        raise ValueError\n    m_in = m_out = 1\n    if 'long_threshold' in p and i > p['long_threshold']:\n        m_in, m_out = p['long_input_mult'], p['long_output_mult']\n    return round(((i - c) * p['input'] * m_in + c * p['cached'] * m_in + o * p['output'] * m_out) / 1e6, 6)\n```",
    ],
    "json_receipts": [
        json.dumps([
            {"store": "Лента-Север", "date": "2026-10-12", "total": 1254.9, "payment": "card"},
            {"store": "Пекарня у моста", "date": "2026-10-13", "total": 500, "payment": "cash"},
            {"store": "Северный путь", "date": "2026-10-14", "total": 3120, "payment": "card"},
            {"store": "Здоровье+", "date": "2026-10-15", "total": 742.5, "payment": "cash"},
        ], ensure_ascii=False),
        json.dumps([
            {"store": "Лента-Север", "date": "2026-10-12", "total": 1254.9, "payment": "card"},
            {"store": "Пекарня у моста", "date": "2026-10-13", "total": 386, "payment": "cash"},
            {"store": "СевНефтеТорг", "date": "2026-10-14", "total": 3120, "payment": "card"},
            {"store": "Здоровье+", "date": "2026-10-15", "total": 742.5, "payment": "cash"},
        ], ensure_ascii=False),
    ],
    "json_model_card": [
        json.dumps({"model_name": "Kestrel-2", "organization": "Larkspur Labs", "total_params_b": 212,
                    "active_params_b": 18, "context_tokens": 128000, "license": "Apache-2.0",
                    "release_date": "2026-09-09"}),
        "not json at all",
    ],
    "logic_pricing": [
        "ОТВЕТ: 666; 360",
        "Подписка дешевле с 667 запросов, экономия 360 долларов.",  # no answer line
    ],
}


@pytest.mark.parametrize("task_id,answer", [(k, a) for k, v in WRONG.items() for a in v])
def test_wrong_answer_fails(task_id, answer):
    assert not check(TASKS[task_id], answer).passed


RIGHT_IN_OTHER_WORDS = {
    "ru_qa": [
        "**1)** 200 × 6,48 + 100 × 3,12 = 1 608 руб.\n**2)** Нет данных.",
        "1) Семья заплатит 1608 рублей.\n2) В тексте нет данных.",
    ],
    "logic_pricing": ["Решение...\n**ОТВЕТ:** 667 запросов; $360"],
    "json_receipts": [
        '{"receipts": [{"store": "Лента Север", "date": "2026-10-12", "total": "1254.90", "payment": "CARD"},'
        '{"store": "Пекарня у моста", "date": "2026-10-13", "total": 386, "payment": "cash"},'
        '{"store": "АЗС Северный путь", "date": "2026-10-14", "total": 3120, "payment": "card"},'
        '{"store": "Аптека Здоровье", "date": "2026-10-15", "total": 742.5, "payment": "cash"}]}',
    ],
    "json_model_card": [
        '{"model_name": "Kestrel 2", "organization": "Larkspur Labs", "total_params_b": 212.0,'
        ' "active_params_b": 18, "context_tokens": 256000, "license": "Apache License 2.0",'
        ' "release_date": "2026-09-09"}',
    ],
    "ru_summary": [
        "1. С 1 апреля 2027 года в центре Верхнедольска самокаты ограничат до 15 км/ч.\n"
        "2. Парковка — только в 120 зонах.\n3. С 23:00 до 6:00 аренда в центре недоступна.",
    ],
}


@pytest.mark.parametrize("task_id,answer", [(k, a) for k, v in RIGHT_IN_OTHER_WORDS.items() for a in v])
def test_right_answer_in_other_words_passes(task_id, answer):
    result = check(TASKS[task_id], answer)
    assert result.passed, result.summary()


def test_code_split_across_blocks():
    t = TASKS["code_bugfix"]
    ref = t.reference()
    first, second = ref.split("\n\n\ndef total_pages")
    resp = (f"Было:\n```python\ndef paginate(items, page, per_page=10):\n    return items\n```\n"
            f"Стало:\n```python\n{first}\n```\n\nИ вторая функция:\n```python\ndef total_pages{second}\n```")
    assert check(t, resp).passed


def test_reasoning_blocks_are_ignored():
    t = TASKS["logic_pricing"]
    assert check(t, "<think>ОТВЕТ: 1; 2</think>\nРешение...\nОТВЕТ: 667; 360").passed


def test_cost_matches_code_cost_reference():
    price = {"input": 2.0, "cached": 0.2, "output": 10.0,
             "long_threshold": 272_000, "long_input_mult": 2.0, "long_output_mult": 1.5}
    assert request_cost(300_000, 10_000, 100_000, price) == 0.97


class _FakeAPI(BaseHTTPRequestHandler):
    """Answers every task with its reference, like a perfect model."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        task = next(t for t in TASKS.values() if t.prompt == prompt)
        payload = {"model": body["model"], "choices": [{"message": {"content": task.reference()}}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 200,
                             "prompt_tokens_details": {"cached_tokens": 100}}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_run_and_report_end_to_end(tmp_path, monkeypatch, capsys):
    server = HTTPServer(("127.0.0.1", 0), _FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import bench.cli as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    try:
        main(["run", "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--model", "fake",
              "--api-key-env", "", "--runs", "2", "--price-input", "2", "--price-output", "10",
              "--price-cached", "0.1"])
    finally:
        server.shutdown()
    out = capsys.readouterr().out
    assert "24/24 passed" in out
    main(["report", str(tmp_path / "results")])
    report = capsys.readouterr().out
    assert "| fake | 12/12 | 24/24 |" in report
    # 900 uncached * $2 + 100 cached * $0.1 + 200 out * $10 = $0.00381 per call, 24 calls
    assert "0.0914" in report


def test_score_manual_answers(tmp_path, monkeypatch, capsys):
    import bench.cli as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    answers = tmp_path / "alice"
    answers.mkdir()
    (answers / "ru_qa.txt").write_text("1) 1608\n2) нет данных", encoding="utf-8")
    (answers / "logic_pricing__run2.txt").write_text("ОТВЕТ: 667; 300", encoding="utf-8")
    main(["score", str(answers), "--label", "alice-ai"])
    out = capsys.readouterr().out
    assert "ru_qa.txt" in out and "PASS" in out
    assert "logic_pricing__run2.txt" in out and "FAIL" in out
    assert "no answers for:" in out
