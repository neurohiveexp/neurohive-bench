#!/usr/bin/env bash
# Checks both scripts without a GPU, llama.cpp or downloads:
#   expert_map.py on a tiny random OLMoE, bench_memory_speed.py on a fake llama-server and nvidia-smi.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP" "$ROOT/results/selftest-good" "$ROOT/results/selftest-weak"' EXIT
cd "$ROOT"

python3 experiments/moe/expert_map.py --random-tiny --out "$TMP/map"
for f in routing.csv expert_share.csv divergence.csv heatmap_ru.png heatmap_en.png heatmap_code.png diff.png; do
  test -s "$TMP/map/$f" || { echo "missing $f"; exit 1; }
done

mkdir -p "$TMP/models"
head -c 300000000 /dev/zero > "$TMP/models/good.gguf"   # sparse files: no disk use on most systems
head -c 120000000 /dev/zero > "$TMP/models/weak.gguf"
FAKE_VRAM="$TMP/vram" python3 experiments/moe/bench_memory_speed.py \
  --llama-server "$HERE/fake-llama-server" --nvidia-smi "$HERE/fake-nvidia-smi" \
  --model "selftest-good=$TMP/models/good.gguf" --model "selftest-weak=$TMP/models/weak.gguf" \
  --runs 1 --out "$TMP/bench"
grep -q "| selftest-good | 0.28 | 1.06 |.*| 12/12 | 100% |" "$TMP/bench/memory_speed.md"
grep -q "| selftest-weak |.*| 70.0 |" "$TMP/bench/memory_speed.md"
echo "selftest OK"
