Extract information about the newly released model from the press release below. Return only a JSON object with these fields:
- "model_name" (string)
- "organization" (string)
- "total_params_b" (number: total parameters, in billions)
- "active_params_b" (number: parameters active per token, in billions, or null if not stated)
- "context_tokens" (integer)
- "license" (SPDX identifier, for example "MIT")
- "release_date" (YYYY-MM-DD)

Press release:

OSLO, September 9, 2026 — Larkspur Labs today released Kestrel-2, the successor to last year's Kestrel-1. Kestrel-2 is a mixture-of-experts model with 212 billion parameters in total, of which 18 billion are active for each token. It supports a context window of 256,000 tokens, double that of Kestrel-1, which was limited to 128,000 tokens and had 140 billion parameters.

Unlike its predecessor, which shipped under the custom Larkspur Community License, Kestrel-2 is available under the Apache License 2.0. The weights can be downloaded today; a hosted API will follow on October 1.

"We trained Kestrel-2 on 31 trillion tokens using our own cluster in Norway," said chief scientist Ingrid Solheim. A smaller dense sibling, Kestrel-2 Mini with 9 billion parameters, is planned for November.
