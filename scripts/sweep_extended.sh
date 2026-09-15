#!/usr/bin/env bash
# Find where the cursive fine-tune actually turns over.
#
# The first sweep's winner sat at the corner of the grid, still improving at its
# final step, which means 1200 steps was a budget limit rather than an optimum
# and carrying it over to the all-Pinkas run carried over a truncation point.
# This runs the winning configuration far past that, plus the two corner
# extensions, so the step count that gets carried over means something.
#
# Held-out split throughout: the 266 test lines are what makes the turning point
# visible at all.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
BASE="${BASE:-runs/final/best.pt}"
OUT=runs/ft_extended
mkdir -p "$OUT"

# Wait for whatever fine-tune is already running, then score it.
while pgrep -f "finetune_hebrew.py" >/dev/null; do sleep 30; done

if [ -f runs/ft_allpinkas/best.pt ]; then
    echo "=== benchmarking the all-Pinkas run ($(date '+%H:%M:%S'))"
    cat > /tmp/ft_all_bench.json <<'JSON'
[
 {"name":"all-pinkas beam12 + charLM + TTA3 + word","kind":"single",
  "checkpoint":"runs/ft_allpinkas/best.pt","beam_width":12,"char_weight":0.4,
  "tta":3,"word_weight":0.2,"pages":true}
]
JSON
    $PY scripts/eval_configs.py --configs /tmp/ft_all_bench.json --pages \
        --json runs/ft_allpinkas_bench.json > runs/ft_allpinkas_bench.log 2>&1
    grep -Ev "^Using|^Found" runs/ft_allpinkas_bench.log | head -4
fi

# ratio 0.5 / lr 5e-5 is the configuration that won; the other two push past the
# corner it sat on, in each direction separately.
for spec in "0.5 5e-5" "0.7 5e-5" "0.5 1e-4"; do
    set -- $spec
    ratio=$1; lr=$2
    tag="r${ratio}_lr${lr}_long"
    echo "=== $tag ($(date '+%H:%M:%S'))"
    $PY scripts/finetune_hebrew.py "$BASE" \
        --corpora pinkas --out "$OUT/$tag" \
        --hebrew-ratio "$ratio" --lr "$lr" \
        --steps 3000 --eval-every 250 \
        > "$OUT/$tag.log" 2>&1
    grep "best held-out" "$OUT/$tag.log" | tail -1
done
echo "=== extended sweep done ($(date '+%H:%M:%S'))"
