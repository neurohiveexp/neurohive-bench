// Тест токенизаторов: один и тот же текст на русском и английском.
// Запуск: node count.mjs  (пакеты: npm i gpt-tokenizer @lenml/tokenizer-gpt4o @lenml/tokenizer-gemma3 @lenml/tokenizer-deepseek_v4 @lenml/tokenizer-qwen3)
import { readFileSync } from "node:fs";
import { encode as o200k } from "gpt-tokenizer/encoding/o200k_base";
import { encode as cl100k } from "gpt-tokenizer/encoding/cl100k_base";
import { fromPreTrained as gemma3 } from "@lenml/tokenizer-gemma3";
import { fromPreTrained as deepseek } from "@lenml/tokenizer-deepseek_v4";
import { fromPreTrained as qwen3 } from "@lenml/tokenizer-qwen3";
import { fromPreTrained as gpt4o } from "@lenml/tokenizer-gpt4o";

const texts = JSON.parse(readFileSync(new URL("./texts.json", import.meta.url)));
const lenml = (t) => (s) => t._encode_text(s).length;
const TOK = {
  "cl100k (GPT-4, GPT-3.5, 2023)": (s) => cl100k(s).length,
  "o200k (GPT-4o / GPT-5)": (s) => o200k(s).length,
  "o200k, проверка через @lenml": lenml(gpt4o()),
  "Gemma 3 (ближайший открытый к Gemini)": lenml(gemma3()),
  "DeepSeek V4": lenml(deepseek()),
  "Qwen3": lenml(qwen3()),
};
const words = (s) => s.split(/\s+/).filter(Boolean).length;
const rows = [];
for (const [tname, f] of Object.entries(TOK)) {
  let ruSum = 0, enSum = 0;
  const per = {};
  for (const [k, v] of Object.entries(texts)) {
    const ru = f(v.ru), en = f(v.en);
    ruSum += ru; enSum += en;
    per[k] = { ru, en, ratio: +(ru / en).toFixed(2), ruPerWord: +(ru / words(v.ru)).toFixed(2), enPerWord: +(en / words(v.en)).toFixed(2) };
  }
  rows.push({ tokenizer: tname, ruTotal: ruSum, enTotal: enSum, ratio: +(ruSum / enSum).toFixed(2), per });
}
for (const [k, v] of Object.entries(texts)) {
  console.log(`# ${k}: RU ${v.ru.length} символов / ${words(v.ru)} слов, EN ${v.en.length} символов / ${words(v.en)} слов`);
}
console.log(JSON.stringify(rows, null, 1));
