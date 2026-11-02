// Тесты калькулятора: node --test tools/cost-calculator/
// 1. Формула из index.html даёт те же результаты, что bench/cost.py, на примерах из tasks/code_cost/tests.py.
// 2. Если есть python3 — сверка с bench/cost.py на случайных входных данных.
// 3. prices.json корректен, а запасная копия в index.html совпадает с ним.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const html = readFileSync(join(here, "index.html"), "utf8");

function block(name) {
  const m = html.match(new RegExp(`/\\* BEGIN ${name}[^*]*\\*/([\\s\\S]*?)/\\* END ${name} \\*/`));
  assert.ok(m, `block ${name} not found in index.html`);
  return m[1];
}
const api = new Function(
  block("cost.js") + block("model.js") +
  "return { requestCost, monthCost, subscriptionBreakEven, gpuBreakEven, SCENARIOS, GPU_FIELDS };"
)();
const { requestCost } = api;

// Те же тарифы и ожидания, что в tasks/code_cost/tests.py
const PLAIN = { input: 2.0, cached: 0.1, output: 10.0 };
const LONG = { input: 2.0, cached: 0.2, output: 10.0, long_threshold: 272_000, long_input_mult: 2.0, long_output_mult: 1.5 };
const close = (a, b) => assert.ok(Math.abs(a - b) < 5e-7, `${a} != ${b}`);

test("plain", () => close(requestCost(100_000, 5_000, 20_000, PLAIN), 0.212));
test("zero", () => assert.equal(requestCost(0, 0, 0, PLAIN), 0));
test("all cached", () => close(requestCost(50_000, 0, 50_000, PLAIN), 0.005));
test("rounding", () => assert.equal(requestCost(1, 1, 0, { input: 1.234567, cached: 0, output: 0 }), 0.000001));
test("long request", () => close(requestCost(300_000, 10_000, 0, LONG), 1.35));
test("long request: cache not multiplied", () => close(requestCost(300_000, 10_000, 100_000, LONG), 0.97));
test("threshold is strict", () => close(requestCost(272_000, 10_000, 0, LONG), 0.644));
test("no long keys in plain", () => close(requestCost(1_000_000, 0, 0, PLAIN), 2.0));
test("null long keys (prices.json format)", () =>
  close(requestCost(1_000_000, 0, 0, { ...PLAIN, long_threshold: null, long_input_mult: null, long_output_mult: null }), 2.0));
test("invalid", () => {
  for (const args of [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [10, 0, 11]]) {
    assert.throws(() => requestCost(...args, PLAIN), RangeError, JSON.stringify(args));
  }
});

test("matches bench/cost.py on random inputs", (t) => {
  const py = spawnSync("python3", ["--version"]);
  if (py.status !== 0) return t.skip("python3 not found");
  let seed = 42;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  const cases = [];
  for (let i = 0; i < 2000; i++) {
    const input = Math.floor(rnd() * 600_000);
    const cached = Math.floor(rnd() * input);
    const output = Math.floor(rnd() * 50_000);
    const price = { input: +(rnd() * 20).toFixed(3), cached: +(rnd() * 2).toFixed(3), output: +(rnd() * 80).toFixed(3) };
    if (rnd() < 0.5) Object.assign(price, { long_threshold: 100_000 + Math.floor(rnd() * 300_000), long_input_mult: 2, long_output_mult: 1.5 });
    cases.push([input, output, cached, price]);
  }
  const script = "import json,sys\nfrom bench.cost import request_cost\n" +
    "print(json.dumps([request_cost(*c) for c in json.load(sys.stdin)]))";
  const r = spawnSync("python3", ["-c", script], { cwd: repo, input: JSON.stringify(cases), encoding: "utf8" });
  assert.equal(r.status, 0, r.stderr);
  const expected = JSON.parse(r.stdout);
  cases.forEach((c, i) => assert.equal(requestCost(...c), expected[i], JSON.stringify(c)));
});

test("prices.json is valid and fallback is in sync", () => {
  const prices = JSON.parse(readFileSync(join(here, "prices.json"), "utf8"));
  assert.ok(prices.models.length > 0);
  for (const m of prices.models) {
    for (const k of ["id", "name", "date", "source"]) assert.equal(typeof m[k], "string", `${m.id}.${k}`);
    for (const k of ["input", "output", "cached"]) assert.ok(m[k] >= 0, `${m.id}.${k}`);
    assert.match(m.date, /^\d{4}-\d{2}-\d{2}$/);
    if (m.long_threshold !== null) {
      assert.ok(m.long_threshold > 0 && m.long_input_mult > 0 && m.long_output_mult > 0, m.id);
    }
    assert.equal(typeof m.verified, "boolean", m.id);
  }
  const fb = html.match(/<script type="application\/json" id="prices-fallback">\n([\s\S]*?)\n<\/script>/);
  assert.ok(fb);
  assert.deepEqual(JSON.parse(fb[1]), prices, "run: node tools/cost-calculator/sync-prices.mjs");
});

test("scenarios produce valid requests", () => {
  const prices = JSON.parse(readFileSync(join(here, "prices.json"), "utf8"));
  for (const s of api.SCENARIOS) {
    const p = Object.fromEntries(s.fields.map((f) => [f[0], f[2]]));
    const d = s.derive(p);
    assert.ok(d.requests > 0 && d.cached <= d.input && d.output > 0, s.id);
    for (const m of prices.models) assert.ok(api.monthCost(d, m).month > 0, `${s.id}/${m.id}`);
  }
  // Разработчик: 2 ч × 40 запросов × 22 дня
  assert.equal(api.SCENARIOS[0].derive({ hours: 2, rph: 40, days: 22, input: 45000, cachePct: 85, output: 1200 }).requests, 1760);
});

test("break-even math", () => {
  assert.equal(api.subscriptionBreakEven(0.1, 20), 200);
  assert.equal(api.subscriptionBreakEven(0, 20), Infinity);
  const g = Object.fromEntries(api.GPU_FIELDS.map((f) => [f[0], f[2]]));
  const d = { input: 10_000, cached: 0, output: 400, requests: 1000, days: 30 };
  const b = api.gpuBreakEven(d, 0.05, g);
  // 4 с на вход + 10 с на выход
  close(b.secondsPerRequest, 14);
  // при пороге экономия на запросах покрывает постоянные расходы
  close(b.requestsPerDay * (0.05 - b.localPerRequest), b.fixedPerDay);
  assert.equal(b.tokensPerDay, b.requestsPerDay * 10_400);
  // если API дешевле электричества, карта не окупается
  assert.equal(api.gpuBreakEven(d, 0, g).requestsPerDay, Infinity);
});
