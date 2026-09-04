"""Score a checkpoint on the ivrit.ai benchmark, in both leaderboard modes.

Prints results next to the published leaderboard so a number is always read in
context. The published figures are the ones the maintainers generated on
2026-09-04 and are baked in below rather than fetched, so an evaluation run is
reproducible offline and cannot silently change under you.

A caution that belongs next to every number this prints: the official harness
(github.com/ivrit-ai/ocr-eval) is private, so `hebocr.metrics` reconstructs the
metric from the leaderboard's description of it. Scoring is maintainer-run, so
the official number for any model is whatever their harness says.
"""

import argparse
import json
from pathlib import Path

from .data.benchmark import load_lines, load_pages
from .metrics import line_report, page_report
from .recognize import Recognizer

# Published leaderboard, generated 2026-09-04 (data/{line,page}.csv on the
# Space ivrit-ai/hebrew-handwriting-ocr-leaderboard).
PUBLISHED_LINE = [
    ("gemini-flash", 0.118824, 212),
    ("gemini-flash-lite", 0.280000, 225),
    ("gpt-5.6-sol", 0.440000, 225),
    ("gpt-5.6-terra", 0.584906, 225),
    ("claude-sonnet-5", 0.615385, 225),
    ("claude-opus-5", 0.692308, 225),
    ("gpt-5.6-luna", 0.734940, 225),
    ("claude-haiku-4-5", 0.904762, 225),
]
PUBLISHED_PAGE = [
    ("gpt-5.6-sol", 0.461852, 0.400282),
    ("gemini-flash-lite", 0.453116, 0.353075),
    ("claude-opus-5", 0.335469, 0.531724),
    ("gpt-5.6-terra", 0.270239, 0.598528),
    ("gemini-flash", 0.215492, 0.763751),
    ("claude-sonnet-5", 0.210250, 0.619099),
    ("gpt-5.6-luna", 0.199185, 1.296665),
    ("claude-haiku-4-5", 0.197437, 0.792767),
]

# Scored here from the benchmark's own second independent transcription of every
# line. Not a model result -- it is what a careful human scores against the
# adjudicated gold, and therefore roughly the floor these metrics can reach.
HUMAN_LINE_CER = 0.000
HUMAN_PAGE_WCOV = 0.890


def _rank_line(cer: float) -> str:
    better = sum(1 for _, published, _ in PUBLISHED_LINE if cer < published)
    return f"{len(PUBLISHED_LINE) - better + 1} of {len(PUBLISHED_LINE) + 1}"


def _rank_page(wcov: float) -> str:
    better = sum(1 for _, published, _ in PUBLISHED_PAGE if wcov > published)
    return f"{len(PUBLISHED_PAGE) - better + 1} of {len(PUBLISHED_PAGE) + 1}"


def evaluate_lines(recognizer: Recognizer, batch_size: int = 12):
    """Line mode: gold crops in, text out."""
    lines = load_lines()
    predictions = recognizer.read([l.image for l in lines], batch_size=batch_size)
    report = line_report([l.text for l in lines], predictions)
    records = [
        {"line_id": l.line_id, "page_id": l.page_id, "gold": l.text, "pred": p}
        for l, p in zip(lines, predictions)
    ]
    return report, records


def evaluate_pages(recognizer: Recognizer, min_confidence: float = 0.0):
    """Full-page mode: whole page in, transcript out."""
    pages = load_pages()
    predictions, records = [], []
    for page in pages:
        transcript, lines = recognizer.read_page(page.image, min_confidence=min_confidence)
        predictions.append(transcript)
        records.append({"page_id": page.page_id, "gold": page.text, "pred": transcript, "lines": lines})
    report = page_report([p.text for p in pages], predictions)
    return report, records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None, help="write predictions and scores here as JSON")
    ap.add_argument("--mode", default="both", choices=["line", "page", "both"])
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop page lines the model is unsure of (0 keeps all)")
    ap.add_argument("--beam-width", type=int, default=0,
                    help="0 = greedy decoding; >1 enables CTC prefix beam search")
    ap.add_argument("--lm", default=None, help="path to a CharNGramLM for shallow fusion")
    ap.add_argument("--lm-weight", type=float, default=0.4)
    ap.add_argument("--length-bonus", type=float, default=0.6)
    args = ap.parse_args()

    lm = None
    if args.lm:
        from .lm import CharNGramLM

        lm = CharNGramLM.load(args.lm)
        print(f"language model: {args.lm}  {lm.stats()}")
        if not args.beam_width:
            print("  note: an LM only applies during beam search; pass --beam-width")

    recognizer = Recognizer(
        args.checkpoint, lm=lm, lm_weight=args.lm_weight,
        beam_width=args.beam_width, length_bonus=args.length_bonus,
    )
    print(f"checkpoint: {args.checkpoint}  (trained {recognizer.trained_epochs} epochs)\n")
    payload: dict = {"checkpoint": str(args.checkpoint)}

    if args.mode in ("line", "both"):
        report, records = evaluate_lines(recognizer, batch_size=args.batch_size)
        payload["line"] = {"report": report.as_dict(), "predictions": records}
        print("LINE MODE  (ranked by median CER, lower is better)")
        print(f"  {report.summary()}")
        print(f"  would rank {_rank_line(report.cer_median)} against the published board\n")
        print(f"  {'model':<22} {'CER median':>10}")
        print(f"  {'-'*22} {'-'*10}")
        print(f"  {'human (2nd read)':<22} {HUMAN_LINE_CER:>10.3f}")
        shown = False
        for name, cer, _ in PUBLISHED_LINE:
            if not shown and report.cer_median < cer:
                print(f"  {'>> THIS MODEL':<22} {report.cer_median:>10.3f}")
                shown = True
            print(f"  {name:<22} {cer:>10.3f}")
        if not shown:
            print(f"  {'>> THIS MODEL':<22} {report.cer_median:>10.3f}")
        print()

    if args.mode in ("page", "both"):
        report, records = evaluate_pages(recognizer, min_confidence=args.min_confidence)
        payload["page"] = {"report": report.as_dict(), "predictions": records}
        print("FULL-PAGE MODE  (ranked by word coverage, higher is better)")
        print(f"  {report.summary()}")
        print(f"  would rank {_rank_page(report.wcov_micro)} against the published board\n")
        print(f"  {'model':<22} {'word cov':>9} {'page CER':>9}")
        print(f"  {'-'*22} {'-'*9} {'-'*9}")
        print(f"  {'human (2nd read)':<22} {HUMAN_PAGE_WCOV:>9.3f} {'-':>9}")
        shown = False
        for name, wcov, cer in PUBLISHED_PAGE:
            if not shown and report.wcov_micro > wcov:
                print(f"  {'>> THIS MODEL':<22} {report.wcov_micro:>9.3f} {report.cer_micro:>9.3f}")
                shown = True
            print(f"  {name:<22} {wcov:>9.3f} {cer:>9.3f}")
        if not shown:
            print(f"  {'>> THIS MODEL':<22} {report.wcov_micro:>9.3f} {report.cer_micro:>9.3f}")
        print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
