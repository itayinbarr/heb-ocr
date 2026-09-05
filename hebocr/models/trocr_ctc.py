"""A pretrained handwriting ViT encoder with our CTC head.

Everything this project achieved so far came from making training ink look more
like real ink: corrected augmentation, lines composed from real handwritten
glyphs, then real Arabic and English handwriting mixed in. Each step helped,
and each was a way of manufacturing a prior that some models simply have.

TrOCR's encoder already has it. It is a ViT pretrained on IAM, real human
handwriting, so it starts out knowing what a pen stroke on paper looks like.
This module keeps our CTC head on top of it, so the properties that made the
CTC model attractive survive: no autoregressive decoder to fall into a
repetition loop, and no subword tokenizer to mishandle right-to-left text or
Hebrew final forms.

Two shape problems have to be solved to put line images through a ViT built for
384x384 photographs:

  Position embeddings. The checkpoint carries 577 of them, laid out as a 24x24
  grid plus a class token. A 64 px line is a 4-row grid of arbitrary width, so
  they are interpolated onto that grid; HuggingFace's ViT does this natively
  through `interpolate_pos_encoding`.

  Time resolution. A patch is 16 px wide, so a raw 64 px-tall line yields only
  width/16 CTC steps. The benchmark's longest line is 111 characters in about
  2433 px, which leaves 152 steps: enough in principle, but CTC also needs a
  blank between repeated characters, so it is uncomfortably tight. The input is
  therefore upscaled horizontally before patching, which halves the effective
  patch stride and doubles the number of steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TROCR_BASE = "microsoft/trocr-base-handwritten"

# ViT was trained on images normalized to [-1, 1].
IMAGENET_MEAN = 0.5
IMAGENET_STD = 0.5


def _blocks(vit) -> nn.ModuleList:
    """The transformer blocks, wherever this transformers version keeps them.

    The attribute moved between releases: `encoder.layer` in transformers 4.x,
    a top-level `layers` in 5.x. Probing rather than assuming means a library
    upgrade cannot silently freeze nothing.
    """
    if hasattr(vit, "layers"):
        return vit.layers
    if hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        return vit.encoder.layer
    raise AttributeError(f"cannot locate transformer blocks on {type(vit).__name__}")


class TrOCREncoderCTC(nn.Module):
    """Pretrained ViT encoder, height collapsed, CTC head on top."""

    def __init__(
        self,
        n_classes: int,
        pretrained: str = TROCR_BASE,
        width_scale: int = 2,
        proj_dim: int = 512,
        dropout: float = 0.1,
        freeze_layers: int = 0,
        grad_checkpointing: bool = True,
        encoder=None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.width_scale = width_scale

        if encoder is None:
            from transformers import VisionEncoderDecoderModel

            encoder = VisionEncoderDecoderModel.from_pretrained(pretrained).encoder
        self.vit = encoder
        # The pooler is a classification head we never use, and it arrives
        # randomly initialized from this checkpoint.
        if hasattr(self.vit, "pooler"):
            self.vit.pooler = None
        if grad_checkpointing:
            self.vit.gradient_checkpointing_enable()

        self.patch = self.vit.config.patch_size
        self.hidden = self.vit.config.hidden_size
        self.rows = 64 // self.patch  # patch rows for a 64 px line

        if freeze_layers:
            # Freezing the lowest blocks keeps the generic stroke and edge
            # detectors intact while the upper blocks adapt to Hebrew.
            for p in self.vit.embeddings.parameters():
                p.requires_grad_(False)
            for layer in _blocks(self.vit)[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Linear(self.hidden * self.rows, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, n_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @property
    def stride(self) -> int:
        """Input pixels per CTC step, after the horizontal upscale."""
        return self.patch // self.width_scale

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths, self.stride, rounding_mode="floor").clamp(min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 1, 64, W) in [0, 1] with ink high -> log-probs (B, W/stride, C)."""
        b, _, h, w = images.shape

        # Undo our ink-high convention: the pretrained encoder expects natural
        # images, dark ink on light paper.
        x = 1.0 - images
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x.repeat(1, 3, 1, 1)

        target_w = max(self.patch, (w * self.width_scale // self.patch) * self.patch)
        if target_w != w:
            x = F.interpolate(x, size=(h, target_w), mode="bilinear", align_corners=False)

        tokens = self.vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        tokens = tokens[:, 1:, :]  # drop the class token

        cols = tokens.shape[1] // self.rows
        # Patches arrive row-major over the (rows x cols) grid. Grouping the
        # vertical patches of one column into a single step keeps full stroke
        # height at every time step rather than pooling it away.
        tokens = tokens.reshape(b, self.rows, cols, self.hidden)
        tokens = tokens.permute(0, 2, 1, 3).reshape(b, cols, self.rows * self.hidden)

        y = self.norm(self.proj(tokens))
        return F.log_softmax(self.head(y), dim=-1)


def build_trocr_ctc(n_classes: int, **kwargs) -> TrOCREncoderCTC:
    return TrOCREncoderCTC(n_classes=n_classes, **kwargs)
