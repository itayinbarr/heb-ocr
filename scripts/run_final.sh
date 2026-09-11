#!/usr/bin/env bash
# The long run: everything available, at the size that fits the card.
#
# Takes the optimizer decision from the A/B rather than assuming it. Pass
# --sam or --no-sam to override; with neither, it reads runs/ab_*/log.jsonl and
# picks whichever arm reached a better benchmark CER, which is the comparison
# scripts/run_sam_ab.sh was built to make.
#
# Data is everything there is: the full DiffusionPen train split, all 551k real
# lines in eight other scripts, all 10,219 real Hebrew lines, and glyph lines
# generated on demand so the synthetic corpus is not bounded by memory.
#
# Detached and cache-only. A run measured in days must not depend on a session
# staying open or a link staying up.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"

PY=.venv/bin/python
INIT=runs/stage_a/best.pt
OUT=runs/final
GLYPHS="${GLYPHS:-2000000}"
EPOCHS="${EPOCHS:-4}"

SAM_FLAG=""
case "${1:-auto}" in
    --sam)    SAM_FLAG="--sam" ;;
    --no-sam) SAM_FLAG="" ;;
    auto)
        VERDICT="$($PY - <<'PYEOF'
import json
from pathlib import Path

def best(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return min((r["benchmark"]["cer_median"] for r in rows), default=None)

nosam, sam = best("runs/ab_nosam/log.jsonl"), best("runs/ab_sam/log.jsonl")
if nosam is None or sam is None:
    print("unknown")
else:
    print("sam" if sam < nosam else "nosam")
PYEOF
)"
        if [ "$VERDICT" = "unknown" ]; then
            echo "the A/B has not finished; pass --sam or --no-sam to decide by hand" >&2
            exit 1
        fi
        [ "$VERDICT" = "sam" ] && SAM_FLAG="--sam"
        echo "[final] A/B says: $VERDICT"
        ;;
    *) echo "usage: $0 [--sam|--no-sam|auto]" >&2; exit 1 ;;
esac

if [ ! -f "$INIT" ]; then
    echo "missing $INIT; the long run starts from the stage A pretrain" >&2
    exit 1
fi

BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)"
if [ -n "$BUSY" ]; then
    echo "the GPU is not free, refusing to start:" >&2
    echo "$BUSY" >&2
    echo "stop those first (mogli-asr.service and llama-server are the usual two)" >&2
    exit 1
fi

echo "[final] starting ${EPOCHS} epochs, ${GLYPHS} generated glyph lines, SAM='${SAM_FLAG:-off}' ($(date '+%F %T'))"
rm -rf "$OUT"
$PY -m hebocr.train --out "$OUT" \
    --size large \
    --init-from "$INIT" \
    --epochs "$EPOCHS" \
    --val-limit 4000 \
    --glyph-lines "$GLYPHS" \
    --endless-glyphs \
    --real-hebrew all \
    --real-ink all \
    --pixel-budget 22000 \
    --concat-prob 0.35 \
    --num-workers 10 \
    --lr 1.5e-4 \
    --ema 0.9995 \
    --seed 0 \
    $SAM_FLAG \
    > "$OUT.log" 2>&1
echo "[final] training finished ($(date '+%F %T'))"

$PY scripts/make_results.py "$OUT/best.pt" --beam-width 12 --lm-weight 0.4 \
    --out RESULTS_final.md --json runs/results_final.json > runs/final_eval.log 2>&1
echo "[final] scored ($(date '+%F %T'))"
tail -20 runs/final_eval.log
