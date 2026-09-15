"""Loading a trained checkpoint and reading line images with it."""

from pathlib import Path

import torch
from PIL import Image

from .charset import Charset
from .data.transforms import LINE_HEIGHT, preprocess
from .decode import beam_decode, greedy_confidence, greedy_decode
from .models.htr_vt import build_model


def _bare_vit(config: dict):
    """A randomly initialized ViT of the right shape, to be filled from the
    checkpoint. Avoids downloading pretrained weights only to discard them."""
    from transformers import ViTConfig, ViTModel

    return ViTModel(
        ViTConfig(
            image_size=config.get("vit_image_size", 384),
            patch_size=config.get("vit_patch_size", 16),
            hidden_size=config.get("vit_hidden", 768),
            num_hidden_layers=config.get("vit_layers", 12),
            num_attention_heads=config.get("vit_heads", 12),
            intermediate_size=config.get("vit_mlp", 3072),
        ),
        add_pooling_layer=False,
    )


class Recognizer:
    """A trained line recognizer, ready to read cropped lines."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device | None = None,
        lm=None,
        lm_weight: float = 0.4,
        beam_width: int = 0,
        length_bonus: float = 0.6,
        tta: int = 0,
        word_lm=None,
        word_weight: float = 0.0,
    ):
        """`beam_width` of 0 means greedy decoding; anything else beam-searches.

        An LM is only consulted during beam search -- there is nothing for it to
        re-rank on a single best path.
        """
        self.lm = lm
        self.lm_weight = lm_weight
        self.beam_width = beam_width
        self.length_bonus = length_bonus
        self.tta = tta
        self.word_lm = word_lm
        self.word_weight = word_weight
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.charset = Charset(chars=state["charset"])
        config = state.get("config", {})
        if config.get("arch") == "trocr":
            from .models.trocr_ctc import build_trocr_ctc

            # The pretrained weights are about to be overwritten by the
            # checkpoint, so skip the download and build it from config.
            self.model = build_trocr_ctc(
                self.charset.n_classes, grad_checkpointing=False, encoder=_bare_vit(config)
            )
        else:
            self.model = build_model(
                self.charset.n_classes,
                config.get("size", "base"),
                mask_ratio=0.0,  # a regularizer; it must not perturb inference
            )
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        # A checkpoint from a fine-tune counts steps, not epochs, and
        # export_model.py carries the missing field through as None rather than
        # dropping it. `.get(k, default)` does not help when the key is present
        # and null, which is how this first crashed on a released file.
        epoch = state.get("epoch")
        self.trained_epochs = (epoch + 1) if isinstance(epoch, int) else 0
        self.trained_steps = state.get("step")

    @torch.no_grad()
    def read(
        self,
        images: list[Image.Image],
        batch_size: int = 12,
        return_confidence: bool = False,
        emitting_only: bool = False,
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
            if self.beam_width and self.beam_width > 1:
                decoded = beam_decode(
                    logprobs, lengths, self.charset,
                    beam_width=self.beam_width, lm=self.lm,
                    lm_weight=self.lm_weight, length_bonus=self.length_bonus,
                    word_lm=self.word_lm, word_weight=self.word_weight,
                )
            else:
                decoded = greedy_decode(logprobs, lengths, self.charset)
            # Confidence always comes from the best path: it is a rejection
            # signal, and a beam's score is not comparable across line lengths.
            scores = greedy_confidence(logprobs, lengths, emitting_only=emitting_only)
            for j, i in enumerate(chunk):
                texts[i] = decoded[j]
                confs[i] = scores[j]

        ordered_texts = [texts[i] for i in range(len(images))]
        if return_confidence:
            return ordered_texts, [confs[i] for i in range(len(images))]
        return ordered_texts

    # Horizontal rescalings used for test-time augmentation. Real hands vary
    # widely in how tightly they pack characters, and the model's reading of a
    # line genuinely changes with that spacing; these bracket the range.
    TTA_SCALES = (1.0, 0.82, 1.22, 0.68, 1.45)

    @torch.no_grad()
    def read_tta(self, images: list[Image.Image], batch_size: int = 12):
        """Read each line at several horizontal scales, keep the most confident.

        Not averaged: CTC outputs at different scales have different numbers of
        time steps, so their logits do not line up and cannot be pooled
        frame-by-frame. Selecting whole hypotheses sidesteps the alignment
        problem entirely.

        The selection score counts only the frames that emit a character. The
        all-frame mean is not comparable between two widths of the same line,
        because the wider one carries more blank frames and blanks score near
        1.0, so it would win on width rather than on legibility.
        """
        n_views = max(1, min(self.tta or 1, len(self.TTA_SCALES)))
        if n_views == 1:
            return self.read(images, batch_size=batch_size)

        best_text = [""] * len(images)
        best_score = [float("-inf")] * len(images)

        for scale in self.TTA_SCALES[:n_views]:
            if scale == 1.0:
                views = images
            else:
                views = [
                    im.resize(
                        (max(8, int(im.width * scale)), im.height), Image.BICUBIC
                    )
                    for im in images
                ]
            texts, confs = self.read(
                views, batch_size=batch_size, return_confidence=True, emitting_only=True
            )
            for i, (text, conf) in enumerate(zip(texts, confs)):
                # An empty read is never preferable to a non-empty one.
                score = conf if text.strip() else float("-inf")
                if score > best_score[i]:
                    best_score[i], best_text[i] = score, text

        return best_text

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

        if self.tta and self.tta > 1:
            texts = self.read_tta(crops)
            _, confs = self.read(crops, return_confidence=True)
        else:
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
