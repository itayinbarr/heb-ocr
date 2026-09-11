"""Read Hebrew handwriting with Mishkefet-v1.

    python predict.py line.jpg                  # one cropped line
    python predict.py page.jpg --page           # a whole page
    python predict.py lines/*.jpg               # several crops
    python predict.py line.jpg --fast           # greedy, no LM, no TTA

The defaults are the configuration that scores best on the ivrit.ai benchmark
for the mode being run, and the two modes do not want the same thing.

Line mode is ranked by median character error rate, and there multi-scale
reading is worth 4 percent relative (0.213 against 0.222) for three forward
passes per line. Full-page mode is ranked by order-independent word coverage,
and there a word prior is worth 0.349 against 0.333 while multi-scale reading is
not the better trade. So lines default to TTA and pages default to the word
prior. `EXPERIMENTS.md` has the measurements.

Everything needed is in this repository: weights, charset, both language models
and the recognizer code. No network access is required at inference time.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hebocr.lm import CharNGramLM          # noqa: E402
from hebocr.recognize import Recognizer    # noqa: E402
from hebocr.wordlm import WordUnigramLM    # noqa: E402


def _find(name: str) -> Path | None:
    """Model files sit next to this script, or under runs/lm in the repo."""
    for candidate in (HERE / name, HERE / "runs" / "lm" / name):
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", help="line crops, or a page with --page")
    ap.add_argument("--page", action="store_true", help="segment a full page first")
    ap.add_argument("--checkpoint", default=None,
                    help="default: mishkefet-v1.pt beside this script")
    ap.add_argument("--fast", action="store_true",
                    help="greedy decoding, no LM and no TTA; faster and worse")
    ap.add_argument("--beam-width", type=int, default=12)
    ap.add_argument("--lm-weight", type=float, default=0.4)
    ap.add_argument("--tta", type=int, default=None,
                    help="horizontal scales to read each line at; default 3 for "
                         "line mode, 0 for page mode")
    ap.add_argument("--word-weight", type=float, default=None,
                    help="word prior strength; default 0.2 for page mode, 0 for "
                         "line mode. Above about 0.5 it starts inventing words")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else _find("mishkefet-v1.pt")
    if checkpoint is None or not checkpoint.exists():
        ap.error("no checkpoint found; pass --checkpoint")

    # Per-mode defaults, overridable. See the module docstring for why they differ.
    tta = args.tta if args.tta is not None else (0 if args.page else 3)
    word_weight = args.word_weight if args.word_weight is not None else (0.2 if args.page else 0.0)
    if args.fast:
        tta, word_weight = 0, 0.0

    char_lm = None
    if not args.fast:
        path = _find("hebrew_char6.pkl")
        if path is None:
            print("warning: hebrew_char6.pkl not found, falling back to plain beam search",
                  file=sys.stderr)
        else:
            char_lm = CharNGramLM.load(path)

    word_lm = None
    if word_weight:
        path = _find("hebrew_words.pkl")
        if path is None:
            print("warning: hebrew_words.pkl not found, continuing without the word prior",
                  file=sys.stderr)
            word_weight = 0.0
        else:
            word_lm = WordUnigramLM.load(path)

    recognizer = Recognizer(
        checkpoint,
        device=args.device,
        lm=char_lm,
        lm_weight=args.lm_weight,
        beam_width=0 if args.fast else args.beam_width,
        tta=tta,
        word_lm=word_lm,
        word_weight=word_weight,
    )

    if args.page:
        for path in args.images:
            transcript, _ = recognizer.read_page(Image.open(path).convert("RGB"))
            print(f"### {path}\n{transcript}\n")
        return 0

    images = [Image.open(p).convert("RGB") for p in args.images]
    texts = recognizer.read_tta(images) if tta > 1 else recognizer.read(images)
    for path, text in zip(args.images, texts):
        print(f"{path}\t{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
