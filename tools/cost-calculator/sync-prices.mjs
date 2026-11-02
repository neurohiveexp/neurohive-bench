// Копирует prices.json в запасной блок внутри index.html.
// Запускать после каждого изменения тарифов: node tools/cost-calculator/sync-prices.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const htmlPath = join(here, "index.html");
const prices = JSON.parse(readFileSync(join(here, "prices.json"), "utf8"));
const html = readFileSync(htmlPath, "utf8");
const re = /(<script type="application\/json" id="prices-fallback">\n)[\s\S]*?(\n<\/script>)/;
if (!re.test(html)) throw new Error("prices-fallback block not found in index.html");
const json = JSON.stringify(prices).replace(/</g, "\\u003c");
writeFileSync(htmlPath, html.replace(re, (_, a, b) => a + json + b));
console.log(`index.html: fallback updated (${prices.models.length} models)`);
