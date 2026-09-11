"""Grid-search decode-time settings on the held-out dev set.

Nothing here ever looks at the benchmark. See `make_devset.py` for why that
matters and how the dev set is built to sit at the same error rate.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from PIL import Image

from hebocr.lm import CharNGramLM
from hebocr.metrics import line_report
from hebocr.wordlm import WordUnigramLM

STAGE_B = "runs/stage_b/best.pt"
V10 = "runs/base/best.pt"
V6 = "runs/base_v6_realink/best.pt"


def load_devset(path):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    images = [Image.frombytes("L", size, blob) for blob, size in data["images"]]
    return images, data["texts"]


def score(name, reader, images, texts, tta=0):
    start = time.time()
    preds = reader.read_tta(images) if tta and tta > 1 else reader.read(images)
    report = line_report(texts, preds)
    took = time.time() - start
    print(
        f"{name:<52} CER {report.cer_median:.4f}  nodrop {report.cer_median_nodrop:.4f}  "
        f"micro {report.cer_micro:.4f}  wcov {report.wcov_micro:.4f}  "
        f"blank {report.n_total - report.n_scored:>2}  [{took:.0f}s]",
        flush=True,
    )
    return {
        "name": name, "cer_median": report.cer_median,
        "cer_median_nodrop": report.cer_median_nodrop, "cer_micro": report.cer_micro,
        "wcov": report.wcov_micro, "blanks": report.n_total - report.n_scored,
        "seconds": took,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default="runs/dev/dev_s2.2.pkl")
    ap.add_argument("--char-lm", default="runs/lm/hebrew_char6.pkl")
    ap.add_argument("--word-lm", default="runs/lm_new/hebrew_words.pkl")
    ap.add_argument("--stage", required=True,
                    choices=["baseline", "word", "tta", "ensemble", "combined"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from hebocr.ensemble import EnsembleRecognizer
    from hebocr.recognize import Recognizer

    images, texts = load_devset(args.dev)
    print(f"dev set {args.dev}: {len(images)} lines\n", flush=True)
    char_lm = CharNGramLM.load(args.char_lm)
    word_lm = WordUnigramLM.load(args.word_lm)
    results = []

    if args.stage == "baseline":
        results.append(score("greedy", Recognizer(STAGE_B, beam_width=0), images, texts))
        results.append(score("beam 12", Recognizer(STAGE_B, beam_width=12), images, texts))
        results.append(score(
            "beam 12 + char LM (shipped)",
            Recognizer(STAGE_B, beam_width=12, lm=char_lm, lm_weight=0.4), images, texts))
        for w in (0.2, 0.3, 0.5, 0.6, 0.8):
            results.append(score(
                f"beam 12 + char LM w={w}",
                Recognizer(STAGE_B, beam_width=12, lm=char_lm, lm_weight=w), images, texts))

    elif args.stage == "word":
        for ww in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
            results.append(score(
                f"beam 12 + char 0.4 + word {ww}",
                Recognizer(STAGE_B, beam_width=12, lm=char_lm, lm_weight=0.4,
                           word_lm=word_lm, word_weight=ww), images, texts))

    elif args.stage == "tta":
        for n in (3, 5):
            results.append(score(
                f"beam 12 + char LM + TTA {n}",
                Recognizer(STAGE_B, beam_width=12, lm=char_lm, lm_weight=0.4, tta=n),
                images, texts, tta=n))

    elif args.stage == "ensemble":
        combos = [
            ("stage_b + v10", [STAGE_B, V10], [1.0, 1.0]),
            ("stage_b + v10 (2:1)", [STAGE_B, V10], [2.0, 1.0]),
            ("stage_b + v10 + v6", [STAGE_B, V10, V6], [1.0, 1.0, 1.0]),
            ("stage_b + v10 + v6 (3:2:1)", [STAGE_B, V10, V6], [3.0, 2.0, 1.0]),
        ]
        for label, ckpts, weights in combos:
            results.append(score(
                f"{label}, greedy",
                EnsembleRecognizer(ckpts, weights=weights, beam_width=0), images, texts))
            results.append(score(
                f"{label}, beam 12 + char LM",
                EnsembleRecognizer(ckpts, weights=weights, beam_width=12,
                                   lm=char_lm, lm_weight=0.4), images, texts))

    elif args.stage == "combined":
        best = json.loads(Path("runs/dev/best.json").read_text())
        ckpts = [STAGE_B, V10, V6][: best["n_models"]]
        weights = best["weights"][: best["n_models"]]
        results.append(score(
            "ensemble + char LM + word LM",
            EnsembleRecognizer(ckpts, weights=weights, beam_width=12, lm=char_lm,
                               lm_weight=0.4, word_lm=word_lm,
                               word_weight=best["word_weight"]), images, texts))
        for n in (3, 5):
            results.append(score(
                f"ensemble + char LM + word LM + TTA {n}",
                EnsembleRecognizer(ckpts, weights=weights, beam_width=12, lm=char_lm,
                                   lm_weight=0.4, word_lm=word_lm,
                                   word_weight=best["word_weight"], tta=n),
                images, texts, tta=n))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
