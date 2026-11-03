# Tokenizer test: Russian vs English

How many more tokens does the same text take in Russian than in English? Four parallel text pairs (`texts.json`), five open tokenizers, counted locally with no API calls. Used in the Neurohive article "Сколько стоит месяц работы с ИИ" (November 2026).

## Run

```bash
npm install
node count.mjs
```

## Texts

| Key | What | Source |
| --- | --- | --- |
| `news` | short tech news | prompt and reference of `tasks/en_translate` |
| `support` | customer and support agent messages | written for this test |
| `report` | quarterly report paragraph | written for this test |
| `code` | Python function, comments and docstring in RU or EN | written for this test |

## Results (2026-11-02)

| Tokenizer | Models | RU / EN tokens | RU extra, all | Text only | Code only |
| --- | --- | --- | --- | --- | --- |
| Gemma 3 | Google open models, closest public relative of Gemini | 698 / 625 | +12% | +14% | +6% |
| o200k | GPT-4o, GPT-5 | 658 / 568 | +16% | +20% | +6% |
| DeepSeek V4 | DeepSeek V4, V4.1 Flash | 759 / 586 | +30% | +37% | +12% |
| Qwen3 | Qwen3, Qwen3.8 | 820 / 604 | +36% | +43% | +16% |
| cl100k | GPT-4, GPT-3.5 (2023) | 1018 / 572 | +78% | +94% | +35% |

Raw numbers: `results.json`.

## Limits

- OpenAI and Google do not say which tokenizer GPT-6.1 Sol and Gemini 4 Argon use; o200k and Gemma 3 are the closest public ones.
- Claude, GigaChat and Alice AI tokenizers are not public and are not measured here.
- `@lenml/tokenizer-*` packages are community repackagings of the Hugging Face tokenizer files. The o200k count was cross-checked with `gpt-tokenizer` and matched exactly.

License: MIT for code, CC BY 4.0 for texts (same as the rest of the repository).
