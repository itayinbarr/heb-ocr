"""Averaging several checkpoints' frame distributions. This does not work.

Kept because the reason it fails is a property of CTC worth knowing, and because
the obvious next idea after "the ensemble helped" is "average the logits", which
is what this is.

The premise was that every HTR-VT preset downsamples width by the same factor of
4 (`CNNFrontEnd.width_stride`), so for one image every checkpoint emits the same
number of frames and frame t of one can be averaged with frame t of another. The
frame *counts* do match. What the frames contain does not.

CTC constrains the order in which a model emits characters and says nothing
about when it emits them. The alignment is latent, and each training run settles
into its own: typically a sharp spike somewhere inside the character with blanks
either side. Measured between this project's two best checkpoints over 40 dev
lines, of the frames where either model emits a character only 19.9% are frames
where both do, and for characters the two agree on, 96% sit at different frames.

So the average lays one model's spike over the other model's blank and flattens
both. Equal weights scored CER 0.697 against 0.250 for the better model alone,
with word coverage 0.010. Skewing to 2:1 recovered to 0.288, which is to say it
improved only as the second model was weighted into irrelevance. No weighting
fixes this, because the defect is not in the weights.

`rover.py` is the method that does work: it votes on finished strings, where the
alignment is computed rather than assumed.

The charset handling below is sound and worth reusing if frame averaging is ever
revisited with checkpoints that *do* share an alignment, such as snapshots of a
single run, where the premise holds because the alignment was learned once.
"""

from pathlib import Path

import torch
from PIL import Image

from .charset import BLANK, Charset
from .data.transforms import preprocess
from .decode import beam_decode, greedy_confidence, greedy_decode
from .models.htr_vt import build_model


class EnsembleRecognizer:
    """Several checkpoints read the same line; their frame distributions merge."""

    def __init__(
        self,
        checkpoints: list[str | Path],
        weights: list[float] | None = None,
        device: str | torch.device | None = None,
        lm=None,
        lm_weight: float = 0.4,
        beam_width: int = 0,
        length_bonus: float = 0.6,
        word_lm=None,
        word_weight: float = 0.0,
        tta: int = 0,
    ):
        if not checkpoints:
            raise ValueError("an ensemble needs at least one checkpoint")
        self.lm = lm
        self.lm_weight = lm_weight
        self.beam_width = beam_width
        self.length_bonus = length_bonus
        self.word_lm = word_lm
        self.word_weight = word_weight
        self.tta = tta

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.weights = list(weights) if weights else [1.0] * len(checkpoints)
        if len(self.weights) != len(checkpoints):
            raise ValueError("one weight per checkpoint, or none at all")
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        self.models, self.charsets = [], []
        for path in checkpoints:
            state = torch.load(path, map_location="cpu", weights_only=False)
            charset = Charset(chars=state["charset"])
            model = build_model(
                charset.n_classes, state.get("config", {}).get("size", "base"), mask_ratio=0.0
            )
            model.load_state_dict(state["model"])
            model.to(self.device).eval()
            self.models.append(model)
            self.charsets.append(charset)

        # The shared alphabet is the intersection, not the union: a character
        # some model cannot emit would be scored by fewer voters than the rest,
        # which quietly reweights the average per character.
        shared = set(self.charsets[0].chars)
        for cs in self.charsets[1:]:
            shared &= set(cs.chars)
        shared.discard("")
        chars = [""] + sorted(shared)
        self.charset = Charset(chars=chars)

        # For each model, the column of its output that feeds each shared index.
        self.gathers = []
        for cs in self.charsets:
            index = {c: i for i, c in enumerate(cs.chars)}
            self.gathers.append(
                torch.tensor([index[c] if c else BLANK for c in chars],
                             dtype=torch.long, device=self.device)
            )

    @torch.no_grad()
    def _frame_logprobs(self, padded: torch.Tensor) -> torch.Tensor:
        """Average every model's distribution over the shared alphabet."""
        amp = self.device.type == "cuda"
        merged = None
        for model, gather, weight in zip(self.models, self.gathers, self.weights):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(padded)
            probs = out.float().exp().index_select(-1, gather)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            merged = probs * weight if merged is None else merged + probs * weight
        return merged.clamp_min(1e-9).log()

    @torch.no_grad()
    def read(self, images: list[Image.Image], batch_size: int = 12, return_confidence: bool = False):
        if not images:
            return ([], []) if return_confidence else []

        order = sorted(range(len(images)), key=lambda i: images[i].width / max(images[i].height, 1))
        texts: dict[int, str] = {}
        confs: dict[int, float] = {}

        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            arrays = [torch.from_numpy(preprocess(images[i])) for i in chunk]
            widths = torch.tensor([a.shape[-1] for a in arrays], dtype=torch.long)
            padded = torch.zeros(len(arrays), 1, arrays[0].shape[-2], int(widths.max()))
            for j, a in enumerate(arrays):
                padded[j, :, :, : a.shape[-1]] = a

            logprobs = self._frame_logprobs(padded.to(self.device))
            lengths = self.models[0].output_lengths(widths)

            if self.beam_width and self.beam_width > 1:
                decoded = beam_decode(
                    logprobs, lengths, self.charset,
                    beam_width=self.beam_width, lm=self.lm, lm_weight=self.lm_weight,
                    length_bonus=self.length_bonus,
                    word_lm=self.word_lm, word_weight=self.word_weight,
                )
            else:
                decoded = greedy_decode(logprobs, lengths, self.charset)
            scores = greedy_confidence(logprobs, lengths)
            for j, i in enumerate(chunk):
                texts[i] = decoded[j]
                confs[i] = scores[j]

        ordered = [texts[i] for i in range(len(images))]
        if return_confidence:
            return ordered, [confs[i] for i in range(len(images))]
        return ordered

    TTA_SCALES = (1.0, 0.82, 1.22, 0.68, 1.45)

    @torch.no_grad()
    def read_tta(self, images: list[Image.Image], batch_size: int = 12):
        """Read at several horizontal scales, keep the most confident reading."""
        n_views = max(1, min(self.tta or 1, len(self.TTA_SCALES)))
        if n_views == 1:
            return self.read(images, batch_size=batch_size)

        best_text = [""] * len(images)
        best_score = [float("-inf")] * len(images)
        for scale in self.TTA_SCALES[:n_views]:
            views = images if scale == 1.0 else [
                im.resize((max(8, int(im.width * scale)), im.height), Image.BICUBIC)
                for im in images
            ]
            texts, confs = self.read(views, batch_size=batch_size, return_confidence=True)
            for i, (text, conf) in enumerate(zip(texts, confs)):
                score = conf if text.strip() else float("-inf")
                if score > best_score[i]:
                    best_score[i], best_text[i] = score, text
        return best_text

    @torch.no_grad()
    def read_page(self, image: Image.Image, min_confidence: float = 0.0, **segment_kwargs):
        from .page.segment import segment_page

        crops, boxes, angle = segment_page(image, **segment_kwargs)
        if not crops:
            return "", []
        if self.tta and self.tta > 1:
            texts = self.read_tta(crops)
            _, confs = self.read(crops, return_confidence=True)
        else:
            texts, confs = self.read(crops, return_confidence=True)
        records = [
            {"text": t, "confidence": c,
             "box": {"x": b.x, "y": b.y, "w": b.w, "h": b.h}, "skew": angle}
            for t, c, b in zip(texts, confs, boxes)
        ]
        kept = [r["text"] for r in records
                if r["text"].strip() and r["confidence"] >= min_confidence]
        return "\n".join(kept), records
