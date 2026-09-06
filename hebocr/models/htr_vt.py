"""A CNN + ViT encoder with a CTC head, after HTR-VT (Pattern Recognition, 2025).

Why this architecture, for this problem specifically:

  - CTC is alignment-free and, unlike an autoregressive decoder, cannot enter a
    repetition loop. On this benchmark that failure is not hypothetical:
    gemini-flash posts a 0.119 median line CER alongside a *micro* CER of 2.19,
    meaning some lines came back many times longer than the truth.
  - Character-level output means no tokenizer, so there is no BPE vocabulary to
    mishandle right-to-left text or Hebrew final letter forms.
  - The CNN front-end is not optional. HTR-VT's ablation removes it and IAM
    validation CER goes from 3.3% to 26.6%. A pure ViT does not have the
    inductive bias to learn strokes from a small corpus.

Regularization is doing heavy lifting here because every training image is
synthetic: span masking over the feature sequence (below) and, optionally, SAM
(see `hebocr/optim.py`), both of which the paper credits for its small-data
results.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection, then optional downsample."""

    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        )
        self.pool = nn.MaxPool2d(stride) if stride != (1, 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)), inplace=True)
        h = self.norm2(self.conv2(h))
        x = F.relu(h + self.skip(x), inplace=True)
        return self.pool(x)


class CNNFrontEnd(nn.Module):
    """Line image -> feature sequence.

    Height is collapsed entirely (64 -> 1) while width is reduced only 4x, so
    each output step covers roughly 4 input pixels. At a 64 px line height that
    is a few steps per character, which is what CTC needs to place a label
    without the sequence being shorter than the transcription.
    """

    def __init__(self, in_ch: int = 1, width: int = 256):
        super().__init__()
        c1, c2, c3 = width // 4, width // 2, width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(c1, c1, stride=(2, 2)),   # 64 x W  -> 32 x W/2
            ConvBlock(c1, c2, stride=(2, 2)),   # 32 x W/2 -> 16 x W/4
            ConvBlock(c2, c2, stride=(2, 1)),   # 16 x W/4 ->  8 x W/4
            ConvBlock(c2, c3, stride=(2, 1)),   #  8 x W/4 ->  4 x W/4
            ConvBlock(c3, c3, stride=(1, 1)),   #  4 x W/4 ->  4 x W/4
        )
        self.out_channels = c3
        self.out_height = 4
        self.width_stride = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, W/4, C*4)"""
        x = self.blocks(self.stem(x))
        b, c, h, w = x.shape
        # Fold the remaining height into the channel dimension rather than
        # pooling it away, which would discard vertical stroke detail.
        return x.permute(0, 3, 1, 2).reshape(b, w, c * h)


class SpanMask(nn.Module):
    """Zero out contiguous spans of the feature sequence during training.

    Masked-image-modeling as a regularizer, applied where it is cheap: on the
    sequence, not the pixels. It forces the encoder to read a character from its
    neighbours, which is exactly the robustness a model trained only on
    synthetic data lacks when it meets a smudged real one.
    """

    def __init__(self, ratio: float = 0.4, span: int = 4):
        super().__init__()
        self.ratio = ratio
        self.span = span

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.ratio <= 0:
            return x
        b, t, _ = x.shape
        n_spans = max(1, int(t * self.ratio / self.span))
        mask = torch.zeros(b, t, device=x.device, dtype=torch.bool)
        starts = torch.randint(0, max(1, t - self.span), (b, n_spans), device=x.device)
        for offset in range(self.span):
            mask.scatter_(1, (starts + offset).clamp(max=t - 1), True)
        return x.masked_fill(mask.unsqueeze(-1), 0.0)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions.

    Deliberately not learned: line widths vary a great deal between the training
    data (synthetic, fixed render) and the benchmark (photographed crops), and a
    learned table generalizes poorly past the lengths it saw.
    """

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class HTRVT(nn.Module):
    """The full recognizer: CNN front-end -> span mask -> ViT encoder -> CTC head."""

    def __init__(
        self,
        n_classes: int,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cnn_width: int = 256,
        mask_ratio: float = 0.4,
        mask_span: int = 4,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.cnn = CNNFrontEnd(in_ch=1, width=cnn_width)
        self.proj = nn.Linear(self.cnn.out_channels * self.cnn.out_height, dim)
        self.mask = SpanMask(ratio=mask_ratio, span=mask_span)
        self.pos = PositionalEncoding(dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """Input pixel widths -> CTC time steps, matching the CNN's stride."""
        return torch.div(widths, self.cnn.width_stride, rounding_mode="floor").clamp(min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> log-probabilities (B, T, n_classes)."""
        x = self.cnn(images)
        x = self.proj(x)
        x = self.mask(x)
        x = self.drop(self.pos(x))
        x = self.encoder(x)
        x = self.norm(x)
        return F.log_softmax(self.head(x), dim=-1)


# Module-level so the CLI can derive its --size choices from it. Keeping the
# two in separate places is how an added preset shipped as an argparse rejection
# that killed a launch and left the GPU idle for seven hours.
MODEL_PRESETS = {
        "small": dict(dim=256, depth=4, heads=4, cnn_width=192),
        "base": dict(dim=384, depth=6, heads=6, cnn_width=256),
        "large": dict(dim=512, depth=8, heads=8, cnn_width=320),
    # Sized for a corpus an order of magnitude larger than the one `base` was
    # chosen for. At ~150k mostly-synthetic lines the model was data limited and
    # extra capacity bought nothing; at 700k with most of it real ink that is no
    # longer obviously true.
    "xl": dict(dim=640, depth=8, heads=10, cnn_width=384),
}


def build_model(n_classes: int, size: str = "base", **overrides) -> HTRVT:
    """Named presets. `small` is the fast iteration model, `base` the default."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size {size!r}, expected one of {sorted(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    return HTRVT(n_classes=n_classes, **cfg)
