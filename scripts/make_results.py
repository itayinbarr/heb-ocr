"""Run the full evaluation suite and write RESULTS.md.

Evaluates a checkpoint in both leaderboard modes, with and without beam search
and LM fusion, and reports every number against the published board and against
the human floor. This is the artifact to hand a maintainer.
"""

import argparse
import json
import time
from pathlib import Path

from hebocr.data.benchmark import load_lines, load_pages
from hebocr.evaluate import (
    HUMAN_LINE_CER, HUMAN_PAGE_WCOV, PUBLISHED_LINE, PUBLISHED_PAGE,
)
from hebocr.metrics import line_report, page_report
from hebocr.recognize import Recognizer


def line_table(rows) -> str:
    """Published line board with our variants slotted in by score."""
    entries = [(f"*human, 2nd read*", HUMAN_LINE_CER, "*225*", True)]
    entries += [(name, cer, str(n), True) for name, cer, n in PUBLISHED_LINE]
    for label, report in rows:
        entries.append((f"**{label}**", report.cer_median, str(report.n_scored), False))
    entries.sort(key=lambda e: e[1])

    out = ["| model | CER median | lines scored |", "|---|---|---|"]
    out += [f"| {n} | {c:.3f} | {s} |" for n, c, s, _ in entries]
    return "\n".join(out)


def page_table(rows) -> str:
    entries = [("*human, 2nd read*", HUMAN_PAGE_WCOV, None)]
    entries += [(name, wcov, cer) for name, wcov, cer in PUBLISHED_PAGE]
    for label, report in rows:
        entries.append((f"**{label}**", report.wcov_micro, report.cer_micro))
    entries.sort(key=lambda e: -e[1])

    out = ["| model | word coverage | page CER |", "|---|---|---|"]
    for name, wcov, cer in entries:
        out.append(f"| {name} | {wcov:.3f} | {'-' if cer is None else f'{cer:.3f}'} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--lm", default="runs/lm/hebrew_char6.pkl")
    ap.add_argument("--beam-width", type=int, default=12)
    ap.add_argument("--lm-weight", type=float, default=0.4)
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--json", default="runs/results.json")
    ap.add_argument("--skip-beam", action="store_true", help="greedy only (much faster)")
    args = ap.parse_args()

    lines, pages = load_lines(), load_pages()
    gold_lines = [l.text for l in lines]
    gold_pages = [p.text for p in pages]

    lm = None
    if args.lm and Path(args.lm).exists() and not args.skip_beam:
        from hebocr.lm import CharNGramLM

        lm = CharNGramLM.load(args.lm)

    variants = [("this model (greedy)", dict(beam_width=0, lm=None))]
    if not args.skip_beam:
        variants.append((f"this model (beam {args.beam_width})", dict(beam_width=args.beam_width, lm=None)))
        if lm is not None:
            variants.append((
                f"this model (beam {args.beam_width} + char LM)",
                dict(beam_width=args.beam_width, lm=lm, lm_weight=args.lm_weight),
            ))

    line_rows, page_rows, payload = [], [], {}
    for label, kwargs in variants:
        print(f"\n=== {label}", flush=True)
        recognizer = Recognizer(args.checkpoint, **kwargs)

        t0 = time.time()
        predictions = recognizer.read([l.image for l in lines])
        report = line_report(gold_lines, predictions)
        line_rows.append((label, report))
        print(f"  line: {report.summary()}  [{time.time()-t0:.0f}s]", flush=True)

        t0 = time.time()
        page_predictions = [recognizer.read_page(p.image)[0] for p in pages]
        preport = page_report(gold_pages, page_predictions)
        page_rows.append((label, preport))
        print(f"  page: {preport.summary()}  [{time.time()-t0:.0f}s]", flush=True)

        payload[label] = {
            "line": report.as_dict(),
            "page": preport.as_dict(),
            "line_predictions": [
                {"line_id": l.line_id, "gold": l.text, "pred": p}
                for l, p in zip(lines, predictions)
            ],
            "page_predictions": [
                {"page_id": p.page_id, "pred": t} for p, t in zip(pages, page_predictions)
            ],
        }

    best_line = min(line_rows, key=lambda r: r[1].cer_median)
    best_page = max(page_rows, key=lambda r: r[1].wcov_micro)

    # The published model our best configuration is closest to, for the
    # blanks-dropped versus no-drop comparison.
    report = best_line[1]
    rival_name, rival_cer, rival_lines = min(
        PUBLISHED_LINE, key=lambda e: abs(e[1] - report.cer_median)
    )
    if report.cer_median < rival_cer and report.cer_median_nodrop > rival_cer:
        verdict = (
            f"So this model wins on the board's own metric and loses on the stricter "
            f"one. {rival_name} returned text for all {rival_lines} lines, so its "
            f"no-drop median is also {rival_cer:.3f}; ours is inflated by "
            f"{report.n_blank} dropped line(s). Treat the two as a tie until the "
            f"blanks are recovered."
        )
    elif report.cer_median_nodrop < rival_cer:
        verdict = (
            f"This model is ahead of {rival_name} on both readings, so the ranking "
            f"does not depend on the blank-dropping rule."
        )
    else:
        verdict = (
            f"This model is behind {rival_name} on both readings; the blank-dropping "
            f"rule is not what separates them."
        )

    best_drop = report.cer_median
    best_nodrop = report.cer_median_nodrop
    best_scored = report.n_scored

    doc = f"""# Results

Checkpoint: `{args.checkpoint}`. Benchmark: `ivrit-ai/hebrew-handwriting-ocr-benchmark`
(225 gold lines, 10 pages). Published comparisons are the leaderboard's own
figures, generated 2026-09-04.

The benchmark was never trained on and never used to select a checkpoint;
selection used synthetic validation CER only.

## Line mode

{line_table(line_rows)}

Best configuration: **{best_line[0]}** at {best_line[1].cer_median:.3f} median CER.

Full statistics, including the no-drop median that charges blank outputs the
full 1.0 rather than dropping them from the median as the leaderboard does:

| configuration | CER median | no-drop median | micro CER | word cov | scored |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {label} | {r.cer_median:.3f} | {r.cer_median_nodrop:.3f} | {r.cer_micro:.3f} "
        f"| {r.wcov_micro:.3f} | {r.n_scored}/{r.n_total} |"
        for label, r in line_rows
    ) + f"""

## Full-page mode

{page_table(page_rows)}

Best configuration: **{best_page[0]}** at {best_page[1].wcov_micro:.3f} word coverage.

Segmentation recall (fraction of gold lines the segmenter finds, at 50% area
coverage) is **96.0%**, 216 of 225, measured by `scripts/eval_segmentation.py`.
That is the ceiling any recognizer can reach through this pipeline.

## The blank-dropping caveat, applied to ourselves

The leaderboard drops blank outputs before taking the median. When that rule
flatters someone else it is worth pointing out; when it flatters us it is worth
pointing out twice. Both readings of the best configuration:

| metric | this model | nearest published model |
|---|---|---|
| leaderboard rule (blanks dropped) | {best_drop:.3f} over {best_scored} lines | {rival_name} {rival_cer:.3f} over {rival_lines} |
| no-drop (a blank scores 1.0) | {best_nodrop:.3f} | {rival_cer:.3f} |

{verdict}

## Reading these numbers

The official scores are whatever the maintainers' harness produces: the
leaderboard is maintainer-run, and `ivrit-ai/ocr-eval` is private, so
`hebocr/metrics.py` is a reconstruction from the leaderboard's description of
the metric.

The line-mode median is a forgiving statistic. Blank outputs are dropped before
it is taken, which is how gemini-flash posts 0.119 while its micro CER on the
same run is 2.19. The no-drop column above is the honest counterpart.
"""

    Path(args.out).write_text(doc, encoding="utf-8")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
