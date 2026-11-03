"""Which MoE experts does the router pick for Russian, English and Python code?

Loads a Mixture-of-Experts model with transformers, runs three texts (texts.json: ru, en, code),
records the top-k experts the router selects for every token in every MoE layer and saves:

  routing.csv        one row per (text, token, layer, rank): expert id and routing weight
  expert_share.csv   share of routing slots per (text, layer, expert), %
  divergence.csv     per layer and text pair: Jensen-Shannon divergence and top-expert overlap
  heatmap_<text>.png layer x expert heatmap for each text
  diff.png           pairwise difference heatmaps + per-layer divergence

Runs on CPU. Examples:

  python experiments/moe/expert_map.py --model allenai/OLMoE-1B-7B-0125
  python experiments/moe/expert_map.py --random-tiny          # no download, checks the pipeline
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEXT_KEYS = ("ru", "en", "code")
TEXT_NAMES = {"ru": "Russian", "en": "English", "code": "Python code"}

# Router modules: model.layers.<i>.<mlp|block_sparse_moe|...>.<gate|router>
ROUTER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:mlp|block_sparse_moe|feed_forward|moe|sparse_moe)\.(?:gate|router)$")

# Colors (validated default palette of the dataviz guide): sequential blue, diverging blue-gray-red,
# categorical slots 1-3 for the three text pairs.
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32525"]
PAIR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e1e0d9"


# ---------------------------------------------------------------- model loading

def _cfg(config, *names, default=None):
    for n in names:
        v = getattr(config, n, None)
        if v is not None:
            return v
    return default


def tiny_random_model(texts: dict, seed: int = 0):
    """A 4-layer, 16-expert OLMoE with random weights and a byte-level BPE tokenizer trained on the
    texts themselves. Nothing is downloaded; useful only to check that the pipeline works."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(list(texts.values()), trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>")

    torch.manual_seed(seed)
    config = OlmoeConfig(vocab_size=tok.get_vocab_size(), hidden_size=64, intermediate_size=32,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                         num_experts=16, num_experts_per_tok=2, max_position_embeddings=1024,
                         pad_token_id=None, bos_token_id=None, eos_token_id=tokenizer.eos_token_id)
    model = OlmoeForCausalLM(config).eval()
    # Random init gives near-uniform routers; widen them so the heatmaps show some structure.
    with torch.no_grad():
        for name, mod in model.named_modules():
            if ROUTER_RE.search(name):
                mod.weight.normal_(0, 1.0)
    return model, tokenizer, "random-tiny-olmoe"


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    kw = dict(revision=args.revision, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True, **kw)
    model.to(args.device).eval()
    return model, tokenizer, args.model


# ---------------------------------------------------------------- routing capture

class RouterRecorder:
    """Forward hooks on every router module. Works with routers that return logits (older
    transformers, nn.Linear gate) and with routers that return (logits, weights, indices)."""

    def __init__(self, model, top_k: int, n_experts: int):
        self.top_k, self.n_experts = top_k, n_experts
        self.layers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.handles = []
        for name, mod in model.named_modules():
            m = ROUTER_RE.search(name)
            if m:
                self.handles.append(mod.register_forward_hook(self._hook(int(m.group(1)))))
        if not self.handles:
            raise SystemExit("no router modules found (expected names like model.layers.0.mlp.gate); "
                             "is this a MoE model?")

    def _hook(self, layer: int):
        def fn(_mod, _inp, out):
            self.layers[layer] = self._parse(out)
        return fn

    def _parse(self, out):
        tensors = list(out) if isinstance(out, (tuple, list)) else [out]
        tensors = [t for t in tensors if torch.is_tensor(t)]
        logits = next((t for t in tensors if t.is_floating_point() and t.shape[-1] == self.n_experts), None)
        idx = next((t for t in tensors if not t.is_floating_point() and t.shape[-1] == self.top_k), None)
        probs = None
        if logits is not None:
            logits = logits.reshape(-1, self.n_experts).float()
            probs = torch.softmax(logits, dim=-1)
        if idx is None:  # router returned only logits: reproduce the standard top-k selection
            if probs is None:
                raise RuntimeError(f"cannot interpret router output: {[tuple(t.shape) for t in tensors]}")
            weights, idx = torch.topk(probs, self.top_k, dim=-1)
        else:
            idx = idx.reshape(-1, self.top_k)
            # weight = softmax probability of the chosen expert (comparable across models)
            weights = probs.gather(-1, idx) if probs is not None else torch.full(idx.shape, float("nan"))
        return idx.cpu().numpy(), weights.detach().cpu().numpy()

    def clear(self):
        self.layers = {}

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def route_text(model, tokenizer, recorder, text: str, max_tokens: int, device: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    ids = enc["input_ids"].to(device)
    recorder.clear()
    model(input_ids=ids, attention_mask=enc.get("attention_mask", torch.ones_like(ids)).to(device),
          use_cache=False)
    id_list = ids[0].tolist()
    tokens = [tokenizer.decode([i]) for i in id_list]
    special = set(tokenizer.all_special_ids or [])
    keep = [i for i, t in enumerate(id_list) if t not in special]
    return id_list, tokens, keep, dict(recorder.layers)


# ---------------------------------------------------------------- statistics

def expert_share(layers: dict, keep: list[int], n_experts: int) -> np.ndarray:
    """(n_layers, n_experts): share of routing slots that went to each expert, rows sum to 1."""
    order = sorted(layers)
    share = np.zeros((len(order), n_experts))
    for row, layer in enumerate(order):
        idx, _ = layers[layer]
        counts = np.bincount(idx[keep].ravel(), minlength=n_experts)
        share[row] = counts / max(counts.sum(), 1)
    return share


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits: 0 = identical distributions, 1 = disjoint."""
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return (kl(p, m) + kl(q, m)) / 2


def top_overlap(p: np.ndarray, q: np.ndarray, k: int) -> float:
    """Share of the k most used experts that two texts have in common."""
    a, b = set(np.argsort(-p)[:k]), set(np.argsort(-q)[:k])
    return len(a & b) / k


# ---------------------------------------------------------------- plots

def _style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlesize": 10,
        "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 160,
    })


def _ticks(ax, layer_ids, n_experts):
    step = 1 if n_experts <= 32 else 8
    ax.set_xticks(range(0, n_experts, step))
    ax.set_yticks(range(len(layer_ids)))
    ax.set_yticklabels(layer_ids)
    ax.set_xlabel("expert")
    ax.set_ylabel("layer")


def plot_heatmaps(shares, layer_ids, n_experts, model_name, n_tokens, out_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style(plt)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = max(s.max() for s in shares.values()) * 100  # one scale for all texts: comparable colors
    paths = []
    for key, share in shares.items():
        w = min(14, 3 + n_experts * 0.16)
        fig, ax = plt.subplots(figsize=(w, 1.6 + len(layer_ids) * 0.3), layout="constrained")
        im = ax.imshow(share * 100, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        _ticks(ax, layer_ids, n_experts)
        ax.set_title(f"{TEXT_NAMES[key]}: share of routing slots per expert, % "
                     f"({n_tokens[key]} tokens; uniform = {100 / n_experts:.1f}%)", loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        cb.outline.set_visible(False)
        cb.set_label("%")
        fig.text(0.0, -0.01, model_name, color=INK_MUTED, fontsize=7, ha="left", va="top")
        p = out_dir / f"heatmap_{key}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def plot_diff(shares, layer_ids, n_experts, div_rows, model_name, out_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style(plt)
    cmap = LinearSegmentedColormap.from_list("div", DIV)
    pairs = list(itertools.combinations(shares, 2))
    diffs = {(a, b): (shares[a] - shares[b]) * 100 for a, b in pairs}
    lim = max(np.abs(d).max() for d in diffs.values()) or 1.0

    w = min(16, 4 + n_experts * 0.16)
    fig = plt.figure(figsize=(w, 2.2 + len(layer_ids) * 0.3 * len(pairs) + 2.6), layout="constrained")
    gs = fig.add_gridspec(len(pairs) + 1, 1, height_ratios=[len(layer_ids)] * len(pairs)
                          + [max(6, len(layer_ids) * 0.8)])
    im, hm_axes = None, []
    for row, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[row])
        hm_axes.append(ax)
        im = ax.imshow(diffs[(a, b)], aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
        _ticks(ax, layer_ids, n_experts)
        ax.set_title(f"{TEXT_NAMES[a]} − {TEXT_NAMES[b]}: red = {TEXT_NAMES[a]} uses the expert more",
                     loc="left")
    cb = fig.colorbar(im, ax=hm_axes, shrink=0.6, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("difference in share, percentage points")

    ax = fig.add_subplot(gs[-1])
    markers = ["o", "s", "^"]
    for i, (a, b) in enumerate(pairs):
        ys = [r["jsd_bits"] for r in div_rows if r["pair"] == f"{a}-{b}"]
        ax.plot(layer_ids, ys, color=PAIR_COLORS[i % 3], lw=2, marker=markers[i % 3], ms=6,
                label=f"{TEXT_NAMES[a]} vs {TEXT_NAMES[b]}")
    ax.set_xlabel("layer")
    ax.set_ylabel("JS divergence, bits")
    ax.set_ylim(bottom=0)
    ax.set_xticks(layer_ids)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=3)
    ax.set_title("How different the expert choice is, per layer (0 = same experts, 1 = disjoint)", loc="left")
    fig.text(0.0, -0.005, model_name, color=INK_MUTED, fontsize=7, ha="left", va="top")
    p = out_dir / "diff.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0125",
                   help="Hugging Face id or local folder of a MoE model (OLMoE, Qwen MoE, Mixtral, ...)")
    p.add_argument("--revision", help="model revision (commit hash or tag) for reproducibility")
    p.add_argument("--random-tiny", action="store_true",
                   help="use a tiny random OLMoE instead of --model (no download; checks the pipeline)")
    p.add_argument("--texts", default=str(HERE / "texts.json"), help="JSON with keys ru, en, code")
    p.add_argument("--out", default=None, help="output folder (default: experiments/moe/out/<model>)")
    p.add_argument("--dtype", choices=["bf16", "fp32", "fp16"], default="bf16",
                   help="weights dtype; bf16 halves RAM (OLMoE-1B-7B: ~14 GB bf16, ~28 GB fp32)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-tokens", type=int, default=512, help="truncate each text to this many tokens")
    p.add_argument("--threads", type=int, help="torch CPU threads")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    raw = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    missing = [k for k in TEXT_KEYS if k not in raw]
    if missing:
        raise SystemExit(f"{args.texts}: missing keys {missing}")
    texts = {k: raw[k] for k in TEXT_KEYS}

    if args.random_tiny:
        model, tokenizer, model_name = tiny_random_model(texts)
        args.device = "cpu"
    else:
        model, tokenizer, model_name = load_model(args)
    config = model.config
    n_experts = _cfg(config, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
    top_k = _cfg(config, "num_experts_per_tok", "moe_top_k", "top_k")
    if not n_experts or not top_k:
        raise SystemExit("config has no expert count / top-k; is this a MoE model?")

    out_dir = Path(args.out) if args.out else HERE / "out" / re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = RouterRecorder(model, top_k=top_k, n_experts=n_experts)
    print(f"{model_name}: {n_experts} experts, top-{top_k}, {len(recorder.handles)} MoE layers", file=sys.stderr)

    shares, n_tokens, layer_ids = {}, {}, None
    with (out_dir / "routing.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "token_index", "token_id", "token", "layer", "rank", "expert", "weight"])
        for key, text in texts.items():
            ids, tokens, keep, layers = route_text(model, tokenizer, recorder, text, args.max_tokens, args.device)
            if layer_ids is None:
                layer_ids = sorted(layers)
            for layer in layer_ids:
                idx, wts = layers[layer]
                for t in keep:
                    for r in range(top_k):
                        w.writerow([key, t, ids[t], tokens[t], layer, r + 1, int(idx[t, r]),
                                    f"{wts[t, r]:.4f}"])
            shares[key] = expert_share(layers, keep, n_experts)
            n_tokens[key] = len(keep)
            print(f"  {key:5} {len(keep):4} tokens routed", file=sys.stderr)
    recorder.remove()

    with (out_dir / "expert_share.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "layer", "expert", "share_pct"])
        for key, share in shares.items():
            for row, layer in enumerate(layer_ids):
                for e in range(n_experts):
                    w.writerow([key, layer, e, f"{share[row, e] * 100:.3f}"])

    k_top = max(top_k, 1)
    div_rows = []
    for a, b in itertools.combinations(TEXT_KEYS, 2):
        for row, layer in enumerate(layer_ids):
            div_rows.append({"pair": f"{a}-{b}", "layer": layer,
                             "jsd_bits": round(jsd(shares[a][row], shares[b][row]), 4),
                             f"top{k_top}_overlap": round(top_overlap(shares[a][row], shares[b][row], k_top), 3)})
    with (out_dir / "divergence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(div_rows[0]))
        w.writeheader()
        w.writerows(div_rows)

    plot_heatmaps(shares, layer_ids, n_experts, model_name, n_tokens, out_dir)
    plot_diff(shares, layer_ids, n_experts, div_rows, model_name, out_dir)

    summary = {"model": model_name, "n_experts": n_experts, "top_k": top_k, "moe_layers": layer_ids,
               "tokens": n_tokens, "mean_jsd_bits": {}}
    for a, b in itertools.combinations(TEXT_KEYS, 2):
        vals = [r["jsd_bits"] for r in div_rows if r["pair"] == f"{a}-{b}"]
        summary["mean_jsd_bits"][f"{a}-{b}"] = round(sum(vals) / len(vals), 4)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nmean JS divergence across layers (bits, 0 = same experts, 1 = disjoint):")
    for pair, v in summary["mean_jsd_bits"].items():
        print(f"  {pair:8} {v:.3f}")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
