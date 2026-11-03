"""Memory, speed and task quality of GGUF models under llama.cpp server, one model at a time.

For each model the script:
  1. starts llama-server with the GGUF file and waits until it has loaded;
  2. polls nvidia-smi in the background for peak GPU memory (the server process and total delta);
  3. measures prompt processing and generation speed (tokens/s) from llama.cpp's own timings
     on a fixed synthetic prompt;
  4. runs the tasks from tasks/ through `python -m bench run` against the server's
     OpenAI-compatible endpoint;
  5. stops the server and writes one row: memory - speed - share of solved tasks.

Output: <out>/memory_speed.csv, <out>/memory_speed.md, server logs and bench results
(results/<label>/*.jsonl, readable by `python -m bench report`).

Example:
  python experiments/moe/bench_memory_speed.py \\
      --model qwen3-30b-a3b-q4=/models/Qwen3-30B-A3B-Q4_K_M.gguf \\
      --model gpt-oss-20b=/models/gpt-oss-20b-mxfp4.gguf \\
      --llama-server ~/llama.cpp/build/bin/llama-server --runs 3
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

FILLER = ("The city council of Verkhnedolsk met on Tuesday to discuss the transport budget, "
          "new buses, the bridge over the Dolka river and parking fees in the centre. ")


# ---------------------------------------------------------------- helpers

def _safe(name: str) -> str:  # same rule as bench.cli._safe, so we find bench's results folder
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _http(method: str, url: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8") or "null")


def parse_models(specs: list[str], config: str | None) -> list[dict]:
    models = []
    if config:
        for m in json.loads(Path(config).read_text(encoding="utf-8")):
            models.append({"label": m.get("label") or Path(m["path"]).stem, "path": m["path"],
                           "args": m.get("args", "")})
    for spec in specs or []:
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = Path(spec).stem, spec
        models.append({"label": label, "path": path, "args": ""})
    if not models:
        raise SystemExit("no models: pass --model LABEL=PATH.gguf (repeatable) or --config models.json")
    for m in models:
        if not Path(m["path"]).exists():
            raise SystemExit(f"{m['label']}: file not found: {m['path']}")
    labels = [m["label"] for m in models]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate labels: {labels}")
    return models


# ---------------------------------------------------------------- GPU memory

class VramMonitor:
    """Polls nvidia-smi; keeps the peak of (a) memory used by the server's processes and
    (b) total used memory minus the baseline measured before the server started."""

    def __init__(self, nvidia_smi: str | None, gpus: str | None, interval: float):
        self.cmd = shutil.which(nvidia_smi) if nvidia_smi else None
        self.gpus, self.interval = gpus, interval
        self.pids: set[int] = set()
        self.baseline = self.peak_total = self.peak_proc = None
        self._stop = threading.Event()
        self._thread = None

    @property
    def available(self) -> bool:
        return self.cmd is not None

    def _query(self, what: str) -> list[list[str]]:
        cmd = [self.cmd, f"--query-{what}", "--format=csv,noheader,nounits"]
        if self.gpus:
            cmd.append(f"--id={self.gpus}")
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]

    def total_used(self) -> float | None:
        try:
            rows = self._query("gpu=memory.used")
            return sum(float(r[0]) for r in rows) if rows else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def proc_used(self) -> float | None:
        try:
            rows = self._query("compute-apps=pid,used_memory")
        except (OSError, subprocess.SubprocessError):
            return None
        vals = []
        for r in rows:
            try:
                if int(r[0]) in self.pids:
                    vals.append(float(r[1]))
            except (ValueError, IndexError):
                continue  # "[N/A]" on some drivers / inside containers
        return sum(vals) if vals else None

    def start(self, pid: int):
        if not self.available:
            return
        self.baseline = self.total_used()
        self.peak_total = self.peak_proc = None
        self.pids = {pid}
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def sample(self):
        t, p = self.total_used(), self.proc_used()
        if t is not None:
            self.peak_total = t if self.peak_total is None else max(self.peak_total, t)
        if p is not None:
            self.peak_proc = p if self.peak_proc is None else max(self.peak_proc, p)

    def stop(self) -> dict:
        if not self.available:
            return {"vram_peak_mib": None, "vram_peak_process_mib": None, "vram_peak_delta_mib": None}
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        self.sample()
        delta = (self.peak_total - self.baseline) if (self.peak_total is not None and self.baseline is not None) else None
        # Prefer the per-process number: it ignores other programs on the GPU. Some drivers and
        # containers do not report per-process memory, then fall back to the total delta.
        peak = self.peak_proc if self.peak_proc is not None else delta
        return {"vram_peak_mib": peak, "vram_peak_process_mib": self.peak_proc, "vram_peak_delta_mib": delta}


def ram_peak_mib(pid: int) -> float | None:
    """Peak resident memory of the server (Linux). Matters for MoE offload (--n-cpu-moe / -ot)."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- llama.cpp server

class LlamaServer:
    def __init__(self, binary: str, model: dict, host: str, port: int, ngl: int, ctx: int,
                 extra: str, log_path: Path):
        self.base = f"http://{host}:{port}"
        self.cmd = [binary, "-m", model["path"], "--host", host, "--port", str(port),
                    "-ngl", str(ngl), "-c", str(ctx)] \
            + shlex.split(extra) + shlex.split(model.get("args") or "")
        self.log_path = log_path
        self.proc = None

    def start(self):
        self.log = self.log_path.open("w", encoding="utf-8")
        self.log.write("$ " + shlex.join(self.cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        return self.proc.pid

    def wait_ready(self, timeout: float) -> float:
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                status, _ = _http("GET", self.base + "/health", timeout=5)
                if status == 200:
                    return time.monotonic() - started
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"llama-server not ready after {timeout:.0f}s; see {self.log_path}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "log", None):
            self.log.close()


def speed_test(base: str, prompt_tokens: int, gen_tokens: int, runs: int) -> dict:
    """Prompt processing and generation speed from llama.cpp's own timings (no HTTP overhead).
    The prompt is sent as exactly `prompt_tokens` token ids; prompt cache is off for every run."""
    _, tok = _http("POST", base + "/tokenize", {"content": FILLER * (prompt_tokens // 20 + 10)})
    ids = tok["tokens"][:prompt_tokens]
    if len(ids) < prompt_tokens:
        raise RuntimeError(f"filler gave only {len(ids)} tokens")
    body = {"prompt": ids, "n_predict": gen_tokens, "temperature": 0, "ignore_eos": True,
            "cache_prompt": False, "stream": False}
    _http("POST", base + "/completion", dict(body, n_predict=8))  # warm-up, not counted
    pp, tg = [], []
    for _ in range(runs):
        _, r = _http("POST", base + "/completion", body)
        t = r.get("timings") or {}
        pp.append(t.get("prompt_per_second"))
        tg.append(t.get("predicted_per_second"))
    pp = [x for x in pp if x]
    tg = [x for x in tg if x]
    return {"prompt_tok_s": round(statistics.median(pp), 1) if pp else None,
            "gen_tok_s": round(statistics.median(tg), 1) if tg else None}


# ---------------------------------------------------------------- bench

def run_bench(base: str, label: str, runs: int, tasks: list[str] | None, max_tokens: int | None,
              timeout: int, log_path: Path) -> dict:
    res_dir = ROOT / "results" / _safe(label)
    before = set(res_dir.glob("*.jsonl")) if res_dir.exists() else set()
    cmd = [sys.executable, "-m", "bench", "run", "--base-url", base + "/v1", "--model", label,
           "--label", label, "--api-key-env", "", "--runs", str(runs), "--timeout", str(timeout)]
    if max_tokens:
        cmd += ["--max-tokens", str(max_tokens)]
    if tasks:
        cmd += ["--tasks", *tasks]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    new = sorted(set(res_dir.glob("*.jsonl")) - before) if res_dir.exists() else []
    if not new:
        raise RuntimeError(f"bench wrote no results; see {log_path}")
    records = [json.loads(line) for line in new[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
    return summarize_records(records, n_tasks=_count_tasks(tasks)) | {"results_file": str(new[-1].relative_to(ROOT))}


def _count_tasks(patterns):
    sys.path.insert(0, str(ROOT))
    from bench.tasks import load_tasks
    return len(load_tasks(patterns))


def summarize_records(records: list[dict], n_tasks: int) -> dict:
    """Same definition as `bench report`: a task is solved if it passed in the majority of runs.
    Tasks with no answer at all (API errors) count as not solved."""
    per_task: dict[str, list[bool]] = {}
    for r in records:
        per_task.setdefault(r["task_id"], []).append(bool(r["passed"]))
    solved = sum(1 for v in per_task.values() if sum(v) * 2 > len(v))
    out_tok = sum((r.get("usage") or {}).get("output_tokens", 0) for r in records)
    lat = sum(r.get("latency_s") or 0 for r in records)
    return {"tasks_solved": solved, "tasks_total": n_tasks,
            "solved_share": round(solved / n_tasks, 3) if n_tasks else None,
            "runs_passed": sum(r["passed"] for r in records), "runs_total": len(records),
            "bench_tok_s": round(out_tok / lat, 1) if lat else None}


# ---------------------------------------------------------------- report

COLUMNS = ["label", "file", "file_gib", "load_s", "vram_peak_gib", "ram_peak_gib", "prompt_tok_s",
           "gen_tok_s", "bench_tok_s", "tasks_solved", "tasks_total", "solved_share", "runs_passed",
           "runs_total", "vram_peak_process_mib", "vram_peak_delta_mib", "results_file", "error"]


def _gib(mib):
    return round(mib / 1024, 2) if mib is not None else None


def write_report(rows: list[dict], out_dir: Path, meta: dict) -> str:
    with (out_dir / "memory_speed.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v, suffix=""):
        return "—" if v is None or v == "" else f"{v}{suffix}"
    lines = [f"llama.cpp, {meta['date']}, GPU: {meta['gpu'] or 'not detected'}, "
             f"ctx {meta['ctx']}, -ngl {meta['ngl']}, speed: {meta['pp']} prompt / {meta['tg']} generated tokens, "
             f"bench runs: {meta['runs']}", "",
             "| Model | File, GiB | Peak VRAM, GiB | Peak RAM, GiB | Prompt, tok/s | Generation, tok/s "
             "| Tasks solved | Share |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        solved = (f"{r['tasks_solved']}/{r['tasks_total']}" if r.get("tasks_solved") is not None else "—")
        share = f"{r['solved_share'] * 100:.0f}%" if r.get("solved_share") is not None else "—"
        lines.append(f"| {r['label']} | {fmt(r.get('file_gib'))} | {fmt(r.get('vram_peak_gib'))} | "
                     f"{fmt(r.get('ram_peak_gib'))} | {fmt(r.get('prompt_tok_s'))} | {fmt(r.get('gen_tok_s'))} | "
                     f"{solved} | {share} |")
    errors = [f"- {r['label']}: {r['error']}" for r in rows if r.get("error")]
    if errors:
        lines += ["", "Errors:", *errors]
    md = "\n".join(lines) + "\n"
    (out_dir / "memory_speed.md").write_text(md, encoding="utf-8")
    return md


def gpu_name(nvidia_smi: str | None, gpus: str | None) -> str | None:
    cmd = shutil.which(nvidia_smi) if nvidia_smi else None
    if not cmd:
        return None
    try:
        args = [cmd, "--query-gpu=name,memory.total", "--format=csv,noheader"] + ([f"--id={gpus}"] if gpus else [])
        out = subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
        return "; ".join(out.splitlines()) or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", metavar="[LABEL=]PATH.gguf",
                   help="GGUF model; repeat for several. For split GGUF give the first part (-00001-of-...)")
    p.add_argument("--config", help='JSON list: [{"label": ..., "path": ..., "args": "--n-cpu-moe 24"}]')
    p.add_argument("--llama-server", default="llama-server", help="path to the llama-server binary")
    p.add_argument("--server-args", default="", help='extra llama-server args for all models, e.g. "-fa on"')
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--ngl", type=int, default=999, help="layers on GPU (default: all)")
    p.add_argument("--ctx", type=int, default=8192, help="context size; also affects VRAM (KV cache)")
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--nvidia-smi", default="nvidia-smi")
    p.add_argument("--gpus", help="GPU indices for nvidia-smi, e.g. 0 or 0,1 (default: all)")
    p.add_argument("--poll", type=float, default=0.25, help="nvidia-smi poll interval, s")
    p.add_argument("--speed-prompt", type=int, default=512, help="prompt tokens for the speed test")
    p.add_argument("--speed-gen", type=int, default=128, help="generated tokens for the speed test")
    p.add_argument("--speed-runs", type=int, default=3, help="speed test repeats (median)")
    p.add_argument("--runs", type=int, default=3, help="bench runs per task")
    p.add_argument("--tasks", nargs="*", help="task id patterns (default: all tasks)")
    p.add_argument("--max-tokens", type=int, default=4096, help="bench answer limit (reasoning models need more)")
    p.add_argument("--timeout", type=int, default=900, help="bench request timeout, s")
    p.add_argument("--skip-bench", action="store_true", help="memory and speed only")
    p.add_argument("--out", help="output folder (default: experiments/moe/out/bench-<timestamp>)")
    args = p.parse_args(argv)

    models = parse_models(args.model, args.config)
    if not shutil.which(args.llama_server):
        raise SystemExit(f"llama-server not found: {args.llama_server} (pass --llama-server PATH)")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else HERE / "out" / f"bench-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vram = VramMonitor(args.nvidia_smi, args.gpus, args.poll)
    if not vram.available:
        print(f"warning: {args.nvidia_smi} not found; VRAM will not be measured", file=sys.stderr)
    meta = {"date": dt.date.today().isoformat(), "gpu": gpu_name(args.nvidia_smi, args.gpus), "ctx": args.ctx,
            "ngl": args.ngl, "pp": args.speed_prompt, "tg": args.speed_gen,
            "runs": 0 if args.skip_bench else args.runs, "server_args": args.server_args}

    rows = []
    for m in models:
        print(f"\n== {m['label']}  ({m['path']})", file=sys.stderr)
        row = {"label": m["label"], "file": Path(m["path"]).name,
               "file_gib": round(_model_size(m["path"]) / 2**30, 2), "error": ""}
        server = LlamaServer(args.llama_server, m, args.host, args.port, args.ngl, args.ctx,
                             args.server_args, out_dir / f"server_{_safe(m['label'])}.log")
        pid = server.start()
        vram.start(pid)
        try:
            row["load_s"] = round(server.wait_ready(args.startup_timeout), 1)
            print(f"   loaded in {row['load_s']}s", file=sys.stderr)
            row |= speed_test(server.base, args.speed_prompt, args.speed_gen, args.speed_runs)
            print(f"   prompt {row['prompt_tok_s']} tok/s, generation {row['gen_tok_s']} tok/s", file=sys.stderr)
            if not args.skip_bench:
                row |= run_bench(server.base, m["label"], args.runs, args.tasks, args.max_tokens,
                                 args.timeout, out_dir / f"bench_{_safe(m['label'])}.log")
                print(f"   solved {row['tasks_solved']}/{row['tasks_total']}", file=sys.stderr)
        except Exception as e:  # keep going with the next model, record the error
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"   ERROR {row['error']}", file=sys.stderr)
        finally:
            ram = ram_peak_mib(pid)
            mem = vram.stop()
            server.stop()
        row |= mem
        row["vram_peak_gib"] = _gib(mem["vram_peak_mib"])
        row["ram_peak_gib"] = _gib(ram)
        rows.append(row)
        print(f"   peak VRAM {row['vram_peak_gib']} GiB, peak RAM {row['ram_peak_gib']} GiB", file=sys.stderr)
        write_report(rows, out_dir, meta)  # after every model: partial results survive a crash

    (out_dir / "meta.json").write_text(json.dumps(meta | {"models": models}, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(write_report(rows, out_dir, meta))
    print(f"written to {out_dir}", file=sys.stderr)


def _model_size(path: str) -> int:
    """Size of a GGUF, including all parts of a split model (name-00001-of-00003.gguf)."""
    p = Path(path)
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", p.name)
    if m:
        return sum(q.stat().st_size for q in p.parent.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf"))
    return p.stat().st_size


if __name__ == "__main__":
    main()
