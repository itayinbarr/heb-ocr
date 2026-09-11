"""Score a fixed list of decode configurations on the benchmark, once.

The list is pre-registered: it is written down before the benchmark is touched,
and every entry in it is reported whether it wins or loses. The choice among
them was made on the dev set (`make_devset.py`), so the benchmark is measuring
these configurations rather than selecting between them.

Reporting only the winner would turn the benchmark into a selection set through
the back door, which is the thing this project has avoided from the start.
"""

import argparse
import json
import time
from pathlib import Path

from hebocr.data.benchmark import load_lines, load_pages
from hebocr.lm import CharNGramLM
from hebocr.metrics import line_report, page_report
from hebocr.wordlm import WordUnigramLM

STAGE_B = "runs/stage_b/best.pt"
V10 = "runs/base/best.pt"
V6 = "runs/base_v6_realink/best.pt"


class RoverReader:
    """Several recognizers reading the same lines, combined by string vote."""

    def __init__(self, members, weights):
        self.members = members
        self.weights = weights

    def read(self, images, batch_size: int = 12):
        from hebocr.rover import rover_batch

        outputs = [m.read(images, batch_size=batch_size) for m in self.members]
        return rover_batch([list(row) for row in zip(*outputs)], self.weights)

    def read_tta(self, images, batch_size: int = 12):
        from hebocr.rover import rover_batch

        outputs = [m.read_tta(images, batch_size=batch_size) for m in self.members]
        return rover_batch([list(row) for row in zip(*outputs)], self.weights)

    def read_page(self, image, min_confidence: float = 0.0, **segment_kwargs):
        """Segment once, with the pivot, then vote on each line's reading.

        Re-segmenting per member would give the members different line crops and
        there would be nothing to vote on.
        """
        from hebocr.page.segment import segment_page
        from hebocr.rover import rover_batch

        crops, boxes, angle = segment_page(image, **segment_kwargs)
        if not crops:
            return "", []
        outputs = [m.read(crops) for m in self.members]
        texts = rover_batch([list(row) for row in zip(*outputs)], self.weights)
        _, confs = self.members[0].read(crops, return_confidence=True)
        records = [
            {"text": t, "confidence": c,
             "box": {"x": b.x, "y": b.y, "w": b.w, "h": b.h}, "skew": angle}
            for t, c, b in zip(texts, confs, boxes)
        ]
        kept = [r["text"] for r in records
                if r["text"].strip() and r["confidence"] >= min_confidence]
        return "\n".join(kept), records


def build(spec, char_lm, word_lm):
    """A config dict -> a reader."""
    kind = spec.get("kind", "single")
    common = dict(
        beam_width=spec.get("beam_width", 0),
        lm=char_lm if spec.get("char_weight") else None,
        lm_weight=spec.get("char_weight", 0.0),
        word_lm=word_lm if spec.get("word_weight") else None,
        word_weight=spec.get("word_weight", 0.0),
        tta=spec.get("tta", 0),
    )
    if kind == "ensemble":
        from hebocr.ensemble import EnsembleRecognizer

        return EnsembleRecognizer(spec["checkpoints"], weights=spec.get("weights"), **common)

    from hebocr.recognize import Recognizer

    if kind == "rover":
        members = [Recognizer(c, **common) for c in spec["checkpoints"]]
        return RoverReader(members, spec.get("weights"))

    return Recognizer(spec["checkpoint"], **common)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", required=True, help="JSON file of pre-registered configs")
    ap.add_argument("--char-lm", default="runs/lm/hebrew_char6.pkl")
    ap.add_argument("--word-lm", default="runs/lm_new/hebrew_words.pkl")
    ap.add_argument("--json", default="runs/benchmark_configs.json")
    ap.add_argument("--pages", action="store_true", help="also score full-page mode")
    args = ap.parse_args()

    specs = json.loads(Path(args.configs).read_text())
    char_lm = CharNGramLM.load(args.char_lm)
    word_lm = WordUnigramLM.load(args.word_lm)

    lines = load_lines()
    images, gold = [l.image for l in lines], [l.text for l in lines]
    pages = load_pages() if args.pages else []

    out = []
    for spec in specs:
        name = spec["name"]
        reader = build(spec, char_lm, word_lm)
        start = time.time()
        preds = (
            reader.read_tta(images) if spec.get("tta", 0) > 1 else reader.read(images)
        )
        report = line_report(gold, preds)
        took = time.time() - start
        row = {
            "name": name, "spec": spec,
            "cer_median": report.cer_median,
            "cer_median_nodrop": report.cer_median_nodrop,
            "cer_micro": report.cer_micro,
            "wcov": report.wcov_micro,
            "n_scored": report.n_scored, "n_total": report.n_total,
            "seconds": took,
        }
        print(
            f"{name:<50} CER {report.cer_median:.4f}  nodrop {report.cer_median_nodrop:.4f}  "
            f"micro {report.cer_micro:.4f}  wcov {report.wcov_micro:.4f}  "
            f"scored {report.n_scored}/{report.n_total}  [{took:.0f}s]",
            flush=True,
        )

        if args.pages and spec.get("pages"):
            transcripts = [reader.read_page(p.image)[0] for p in pages]
            preport = page_report([p.text for p in pages], transcripts)
            row["page_wcov"] = preport.wcov_micro
            row["page_cer"] = preport.cer_micro
            print(f"{'':<50} page wcov {preport.wcov_micro:.4f}  page CER {preport.cer_micro:.4f}",
                  flush=True)
        out.append(row)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
