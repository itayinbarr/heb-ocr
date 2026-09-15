#!/usr/bin/env bash
# Two questions, one after the other.
#
# 1. Does anything happen past 3000 steps? The 3000-step curve was flat from
#    2250 and its benchmark result was no better than not fine-tuning at all,
#    so the expectation is no. Run to 6000 and find out rather than assume.
#
# 2. All of Pinkas at the mixture that won. 943 lines is the axis that has
#    actually paid: +266 lines bought 0.0123 benchmark CER while +1550 steps
#    bought nothing. Ratio 0.7 beat 0.5 at every step of the extended sweep.
#    That combination has not been tried.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
BASE=runs/final/best.pt

echo "=== 6000 steps, ratio 0.7, lr 5e-5, held-out split ($(date '+%H:%M:%S'))"
$PY scripts/finetune_hebrew.py "$BASE" --corpora pinkas \
    --out runs/ft_6000 --hebrew-ratio 0.7 --lr 5e-5 \
    --steps 6000 --eval-every 250 > runs/ft_6000.log 2>&1
grep "best held-out" runs/ft_6000.log | tail -1

echo "=== all-Pinkas at ratio 0.7, 1200 steps ($(date '+%H:%M:%S'))"
$PY scripts/finetune_hebrew.py "$BASE" --corpora pinkas \
    --train-split all --fixed-steps 1200 \
    --out runs/ft_all_r07 --hebrew-ratio 0.7 --lr 5e-5 \
    --eval-every 400 > runs/ft_all_r07.log 2>&1
tail -4 runs/ft_all_r07.log | head -2

echo "=== benchmarking both ($(date '+%H:%M:%S'))"
cat > /tmp/six_bench.json <<'JSON'
[
 {"name":"6000-step r0.7","kind":"single","checkpoint":"runs/ft_6000/best.pt",
  "beam_width":12,"char_weight":0.4,"tta":3,"word_weight":0.2,"pages":true},
 {"name":"all-Pinkas r0.7","kind":"single","checkpoint":"runs/ft_all_r07/best.pt",
  "beam_width":12,"char_weight":0.4,"tta":3,"word_weight":0.2,"pages":true}
]
JSON
$PY scripts/eval_configs.py --configs /tmp/six_bench.json --pages \
    --json runs/six_bench.json > runs/six_bench.log 2>&1
grep -Ev "^Using|^Found" runs/six_bench.log
echo "=== done ($(date '+%H:%M:%S'))"
