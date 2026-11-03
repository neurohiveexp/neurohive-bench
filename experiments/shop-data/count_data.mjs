import { readFileSync } from "node:fs";
import { encode as o200k } from "gpt-tokenizer/encoding/o200k_base";
import { fromPreTrained as gemma3 } from "@lenml/tokenizer-gemma3";
import { fromPreTrained as deepseek } from "@lenml/tokenizer-deepseek_v4";
import { fromPreTrained as qwen3 } from "@lenml/tokenizer-qwen3";
const lenml = (t) => (s) => t._encode_text(s).length;
const TOK = { "o200k": (s)=>o200k(s).length, "gemma3": lenml(gemma3()), "deepseek_v4": lenml(deepseek()), "qwen3": lenml(qwen3()) };
const files = ["orders_1000.csv","orders_1000.md","orders_1000.json","events_1000.jsonl","schema.sql","sql_result_200.csv"];
const out = {};
for (const f of files) {
  const s = readFileSync(new URL("./data/"+f, import.meta.url),"utf8"); out[f] = {chars: s.length};
  for (const [k,fn] of Object.entries(TOK)) out[f][k] = fn(s);
}
console.log(JSON.stringify(out,null,1));
