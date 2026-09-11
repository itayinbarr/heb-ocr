#!/usr/bin/env bash
# Emit a line only when something changes or goes wrong.
#
# Written because a launched run was assumed to be a running run, and roughly
# seven hours were lost before anyone checked. Silence here means healthy and
# advancing; any output is a state change worth reading.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

last_state=""
last_pos=""
stall_ticks=0

while true; do
    if pgrep -f "hebocr[.]train --out runs/stage_a" >/dev/null; then
        state="stage_a"; log="runs/stage_a.log"
    elif pgrep -f "hebocr[.]train --out runs/stage_b" >/dev/null; then
        state="stage_b"; log="runs/stage_b.log"
    elif pgrep -f "make_results" >/dev/null; then
        state="evaluating"; log="runs/final_eval.log"
    elif [ -f RESULTS.md ] && [ -f runs/stage_b/best.pt ] && \
         [ runs/stage_b/best.pt -ot RESULTS.md ]; then
        echo "PIPELINE COMPLETE: stage B trained and RESULTS.md written"
        exit 0
    else
        state="idle"; log=""
    fi

    if [ "$state" != "$last_state" ]; then
        echo "state -> $state ($(date '+%H:%M'))"
        last_state="$state"; last_pos=""; stall_ticks=0
    fi

    if [ "$state" = "idle" ]; then
        # Idle is only acceptable briefly, while a chain hands over.
        stall_ticks=$((stall_ticks + 1))
        if [ "$stall_ticks" -ge 3 ]; then
            echo "ALERT: nothing running for 15 minutes and the pipeline is not complete"
            stall_ticks=0
        fi
    elif [ -n "$log" ] && [ -f "$log" ]; then
        pos=$(wc -c < "$log")
        if [ "$pos" = "$last_pos" ]; then
            stall_ticks=$((stall_ticks + 1))
            if [ "$stall_ticks" -ge 4 ]; then
                echo "ALERT: $state log has not grown in 20 minutes"
                stall_ticks=0
            fi
        else
            stall_ticks=0
        fi
        last_pos="$pos"
    fi

    sleep 300
done
