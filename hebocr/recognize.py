"""Loading a trained checkpoint and reading line images with it."""

from pathlib import Path

import torch
from PIL import Image

from .charset import Charset
from .data.transforms import preprocess
from .decode import greedy_confidence, greedy_decode
from .models.htr_vt import build_model


class Recognizer:
    """A trained line recognizer, ready to read cropped lines."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.charset = Charset(chars=state["charset"])
        config = state.get("config", {})
        self.model = build_model(
            self.charset.n_classes,
            config.get("size", "base"),
            mask_ratio=0.0,  # a regularizer; it must not perturb inference
        )
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.trained_epochs = state.get("epoch", -1) + 1

    @torch.no_grad()
    def read(
        self,
        images: list[Image.Image],
        batch_size: int = 12,
        return_confidence: bool = False,
    ):
        """Read line crops. Returns texts, or (texts, confidences).

        Images are grouped by aspect ratio before batching so a batch pads to
        roughly the width it needs; on an 8 GB card a single 2400 px line mixed
        into a batch of short ones is the difference between fitting and not.
        """
        if not images:
            return ([], []) if return_confidence else []

        order = sorted(range(len(images)), key=lambda i: images[i].width / max(images[i].height, 1))
        texts: dict[int, str] = {}
        confs: dict[int, float] = {}
        amp = self.device.type == "cuda"

        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            arrays = [torch.from_numpy(preprocess(images[i])) for i in chunk]
            widths = torch.tensor([a.shape[-1] for a in arrays], dtype=torch.long)
            padded = torch.zeros(len(arrays), 1, arrays[0].shape[-2], int(widths.max()))
            for j, a in enumerate(arrays):
                padded[j, :, :, : a.shape[-1]] = a

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logprobs = self.model(padded.to(self.device))
            logprobs = logprobs.float()

            lengths = self.model.output_lengths(widths)
            decoded = greedy_decode(logprobs, lengths, self.charset)
            scores = greedy_confidence(logprobs, lengths)
            for j, i in enumerate(chunk):
                texts[i] = decoded[j]
                confs[i] = scores[j]

        ordered_texts = [texts[i] for i in range(len(images))]
        if return_confidence:
            return ordered_texts, [confs[i] for i in range(len(images))]
        return ordered_texts

    @torch.no_grad()
    def read_page(self, image: Image.Image, min_confidence: float = 0.0, **segment_kwargs):
        """Segment a full page and read it. Returns (transcript, per-line records).

        Lines are joined with newlines in the segmenter's reading order. Word
        coverage -- the metric the full-page board ranks on -- ignores order
        entirely, so a misordered line still contributes its words.
        """
        from .page.segment import segment_page

        crops, boxes, angle = segment_page(image, **segment_kwargs)
        if not crops:
            return "", []

        texts, confs = self.read(crops, return_confidence=True)
        records = [
            {
                "text": t, "confidence": c,
                "box": {"x": b.x, "y": b.y, "w": b.w, "h": b.h},
                "skew": angle,
            }
            for t, c, b in zip(texts, confs, boxes)
        ]
        kept = [r["text"] for r in records if r["text"].strip() and r["confidence"] >= min_confidence]
        return "\n".join(kept), records
