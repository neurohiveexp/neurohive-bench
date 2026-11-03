# MoE experiments: expert routing, memory and speed

Two scripts for Neurohive's articles about Mixture-of-Experts models:

| Script | Question | Needs |
| --- | --- | --- |
| `expert_map.py` | Which experts does the router pick for Russian, English and Python code? | Python, `torch`, `transformers`, `matplotlib`; CPU is enough |
| `bench_memory_speed.py` | How much GPU memory a GGUF model takes, how fast it reads and writes, and how many base tasks it solves | `llama-server` from llama.cpp, `nvidia-smi`; standard library only |

**Status:** both scripts were checked only on a tiny random model and a fake server (see "Self-test"). The container we wrote them in cannot reach Hugging Face and has no GPU, so there are no real results yet. Run them as described below and publish the outputs with the date and the model revision.

## 1. `expert_map.py`: router map

It runs three texts from `texts.json` (`ru` and `en` are parallel translations of a fictional news item, `code` is a Python snippet). For every token and every MoE layer it records the top-k experts the router chose, by hooking the router modules (`model.layers.N.mlp.gate` and similar).

```bash
pip install torch transformers matplotlib numpy
python experiments/moe/expert_map.py --model allenai/OLMoE-1B-7B-0125
```

- `--model` takes any Hugging Face id or local folder with a MoE model in transformers: OLMoE, Qwen1.5/2/3-MoE, Mixtral and others whose router is `layers.N.<mlp|block_sparse_moe>.<gate|router>`. Pin `--revision` for a published result.
- Memory: OLMoE-1B-7B has 6.9B parameters, about 14 GB RAM in bf16 (default) or 28 GB with `--dtype fp32`. Three short texts take a few minutes on a laptop CPU.
- Offline: download once with `huggingface-cli download allenai/OLMoE-1B-7B-0125 --local-dir ./olmoe`, then pass `--model ./olmoe`.
- Your own texts: `--texts my.json` with keys `ru`, `en`, `code`. Longer texts give more stable shares. The script processes up to 512 tokens per text (`--max-tokens`).

Output (default `experiments/moe/out/<model>/`):

| File | Contents |
| --- | --- |
| `routing.csv` | one row per text × token × layer × rank: `token`, `expert`, `weight` (softmax probability of that expert) |
| `expert_share.csv` | share of routing slots per text, layer and expert, %. Each layer sums to 100%; a uniform router gives 100/E |
| `divergence.csv` | per layer and text pair: Jensen-Shannon divergence (bits, 0 = same experts, 1 = disjoint) and overlap of the k most used experts |
| `heatmap_ru.png`, `heatmap_en.png`, `heatmap_code.png` | layer × expert heatmaps on one shared color scale |
| `diff.png` | three difference heatmaps (ru − en, ru − code, en − code, percentage points) and JS divergence per layer |
| `summary.json` | model, number of experts, top-k, tokens per text, mean divergence |

Special tokens (BOS/EOS) are excluded from the statistics. Routing depends on context, so the map describes these three texts, not a language in general.

## 2. `bench_memory_speed.py`: memory, speed and quality

The script runs the models one at a time. For each one it:

1. starts `llama-server` with the GGUF file and waits for `/health`;
2. polls `nvidia-smi` every 0.25 s. Peak VRAM is the memory of the server process. If the driver does not report per-process memory, the script uses total used memory minus the level before the server started. Both values go to the CSV. Peak RAM is `VmHWM` of the server process, which matters when MoE experts are offloaded to the CPU;
3. measures prompt processing and generation speed (median of 3 runs, 512 prompt tokens, 128 generated tokens, prompt cache off) from llama.cpp's own `timings`;
4. runs all tasks from `tasks/` via `python -m bench run` against `http://127.0.0.1:8090/v1` (3 runs; a task counts as solved if it passed in the majority of runs, as in `bench report`);
5. stops the server and writes a row.

```bash
# build llama.cpp with CUDA once
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
```

```bash
python experiments/moe/bench_memory_speed.py \
  --llama-server ~/llama.cpp/build/bin/llama-server \
  --model qwen3-30b-a3b-q4=/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --model gpt-oss-20b=/models/gpt-oss-20b-mxfp4.gguf \
  --runs 3
```

Model names and files are examples. Write down the exact file, quantization and llama.cpp commit (`llama-server --version`) on the test day.

- `--model LABEL=PATH` can be repeated. For split GGUF files, give the first part; the size column adds up all parts.
- Per-model server flags go in a JSON file passed with `--config models.json`. This is useful to compare full offload with MoE experts on the CPU:

  ```json
  [
    {"label": "qwen3-30b-a3b gpu", "path": "/models/Qwen3-30B-A3B-Q4_K_M.gguf"},
    {"label": "qwen3-30b-a3b cpu-moe", "path": "/models/Qwen3-30B-A3B-Q4_K_M.gguf", "args": "--n-cpu-moe 48"}
  ]
  ```

- Flags for all models: `--server-args "-fa on"`, `--ctx 8192` (the KV cache is part of VRAM), `--ngl 999`, `--gpus 0`.
- Reasoning models write long answers: raise `--max-tokens` (default 4096) and `--timeout`.
- `--skip-bench` measures only memory and speed.
- Without `nvidia-smi` (CPU, Apple Silicon) the VRAM column is `—`; speed and tasks are still measured.

Output (default `experiments/moe/out/bench-<timestamp>/`): `memory_speed.csv` (all columns), `memory_speed.md` (summary table), `server_<label>.log`, `bench_<label>.log`, `meta.json`. Answers and rule details are in `results/<label>/*.jsonl`, the same as `python -m bench run`, so `python -m bench report` works on them. The table is rewritten after each model, so partial results survive a crash.

The table looks like this. Numbers are from the self-test with a fake server, not real measurements:

| Model | File, GiB | Peak VRAM, GiB | Peak RAM, GiB | Prompt, tok/s | Generation, tok/s | Tasks solved | Share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selftest-good | 0.28 | 1.06 | 0.07 | 900.0 | 40.0 | 12/12 | 100% |

## Self-test without GPU or downloads

```bash
pip install torch transformers matplotlib numpy
bash experiments/moe/selftest/selftest.sh
```

The self-test does two things:

- It runs `expert_map.py --random-tiny`: a 4-layer, 16-expert OLMoE with random weights and a byte-level BPE tokenizer trained on the three texts. The hooked expert choices match `output_router_logits` exactly. The heatmaps of a random model show no real specialization; they only show that the pipeline works.
- It runs `bench_memory_speed.py` against `selftest/fake-llama-server` and `selftest/fake-nvidia-smi`. The fake server answers bench tasks with the reference answers, reports fixed timings and "uses" VRAM in proportion to the file size. The test checks that memory, speed and the solved count reach the table.

The safety note from the main README applies here: code tasks execute model output, so run the bench in a container or VM.

License: MIT for code, CC BY 4.0 for texts, like the rest of the repository.
