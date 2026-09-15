#!/usr/bin/env bash
# Sweep the cursive-only Hebrew fine-tune.
#
# Cursive only, which means Pinkas alone: BiblIA is 100 percent square script
# (Italian, Sephardi, Ashkenazi, no cursive rows in its catalogue) and square is
# the book hand printed Hebrew type was modelled on, so it is the opposite of
# the target. That leaves 677 training lines and 266 held out, which is the
# entire published supply of real Hebrew cursive with line-level transcriptions.
#
# 677 lines against 30M parameters overfits fast, so the sweep is over the two
# knobs that control how hard the fine-tune pulls: how much of each batch is
# real Hebrew, and how large a step it takes. Selection inside each run is on
# the 266 held-out lines, never the benchmark.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
BASE="${BASE:-runs/final/best.pt}"
OUT=runs/ft_sweep
mkdir -p "$OUT"

for ratio in 0.2 0.3 0.5; do
    for lr in 1e-5 2e-5 5e-5; do
        tag="r${ratio}_lr${lr}"
        echo "=== $tag ($(date '+%H:%M:%S'))"
        $PY scripts/finetune_hebrew.py "$BASE" \
            --corpora pinkas \
            --out "$OUT/$tag" \
            --hebrew-ratio "$ratio" \
            --lr "$lr" \
            --steps 1200 \
            --eval-every 150 \
            --batch-lines 16 \
            > "$OUT/$tag.log" 2>&1
        tail -3 "$OUT/$tag.log" | head -1
    done
done
echo "=== sweep done ($(date '+%H:%M:%S'))"
