"""Score an OpenAI vision model on the benchmark, to sanity-check our scorer.

Why this exists. The official harness (github.com/ivrit-ai/ocr-eval) is
private, so `hebocr.metrics` reconstructs the leaderboard metric from the one
sentence describing it. Running a model that is already ON the board and
comparing our median CER to the published one tells us whether that
reconstruction is close.

It is a sanity check, not a prerequisite. Scoring is maintainer-run -- they
evaluate submissions with their harness -- so the official number never depends
on ours. Expect our figure to land near, not exactly on, the published one: the
prompt they used is not public, and the prompt drives the result.

Published line-mode medians to compare against:
    gpt-5.6-sol    0.440      gpt-5.6-terra  0.585      gpt-5.6-luna  0.735

This calls a paid API. 225 images per run; the cheapest tier costs cents.
Requires OPENAI_API_KEY in the environment.
"""

import argparse
import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hebocr.data.benchmark import load_lines
from hebocr.metrics import line_report

PROMPT = (
    "This image is a single line of handwritten Hebrew. "
    "Transcribe it exactly as written, preserving spelling and punctuation. "
    "Reply with the transcription only, no commentary, no quotation marks."
)


def encode(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-5.6-luna", help="an OpenAI vision model id")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N lines")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; refusing to run.")
        return 2

    from openai import OpenAI

    client = OpenAI()
    lines = load_lines()[: args.limit]
    print(f"scoring {args.model} on {len(lines)} lines...", flush=True)

    def transcribe(line) -> str:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{encode(line.image)}"}},
                    ],
                }],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - a failed call is a blank, not a crash
            print(f"  {line.line_id[:8]}: {type(exc).__name__}", flush=True)
            return ""

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        predictions = list(pool.map(transcribe, lines))

    report = line_report([l.text for l in lines], predictions)
    print(f"\n{args.model}: {report.summary()}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "report": report.as_dict(),
                    "predictions": [
                        {"line_id": l.line_id, "gold": l.text, "pred": p}
                        for l, p in zip(lines, predictions)
                    ],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
