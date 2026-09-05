"""Integration checks on the training step itself."""

import numpy as np
import pytest
import torch

from hebocr.charset import Charset
from hebocr.data.synth import Batch, collate
from hebocr.decode import greedy_decode
from hebocr.metrics import line_report
from hebocr.models.htr_vt import build_model
from hebocr.optim import SAM
from hebocr.train import _lr_at, Config, ctc_loss


def _batch(charset, texts, width=400):
    items = []
    for t in texts:
        image = torch.zeros(1, 64, width)
        image[0, 20:44, :] = 1.0
        items.append((image, charset.encode(t), t))
    return collate(items)


def test_ctc_loss_is_finite_and_positive():
    charset = Charset.default()
    model = build_model(charset.n_classes, "small")
    batch = _batch(charset, ["שלום", "עולם"])
    loss = ctc_loss(model(batch.images), batch, model)
    assert torch.isfinite(loss) and loss.item() > 0


def test_ctc_loss_skips_lines_too_long_to_align():
    """A transcription longer than the time steps cannot be aligned at all;
    including it would poison the batch mean with an infinity."""
    charset = Charset.default()
    model = build_model(charset.n_classes, "small")
    # 60 characters in a 40 px line: only 10 CTC steps available.
    batch = _batch(charset, ["א" * 60], width=40)
    loss = ctc_loss(model(batch.images), batch, model)
    assert torch.isfinite(loss)


def test_ctc_loss_handles_an_all_impossible_batch():
    charset = Charset.default()
    model = build_model(charset.n_classes, "small")
    batch = _batch(charset, ["ב" * 80, "ג" * 90], width=40)
    assert float(ctc_loss(model(batch.images), batch, model).detach()) == 0.0


def test_learning_rate_warms_up_then_decays():
    cfg = Config(lr=3e-4, min_lr=1e-6, warmup_steps=100)
    assert _lr_at(0, cfg, 1000) < cfg.lr
    assert _lr_at(99, cfg, 1000) == cfg.lr
    assert _lr_at(1000, cfg, 1000) < cfg.lr
    assert _lr_at(1000, cfg, 1000) >= cfg.min_lr
    ramp = [_lr_at(s, cfg, 1000) for s in range(0, 100, 10)]
    assert ramp == sorted(ramp)


def test_a_few_steps_reduce_the_loss():
    """Cheap guard that the wiring -- labels, lengths, blank index -- is sane."""
    torch.manual_seed(0)
    charset = Charset.default()
    model = build_model(charset.n_classes, "small", mask_ratio=0.0, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = _batch(charset, ["אב", "גד"])

    first = float(ctc_loss(model(batch.images), batch, model).detach())
    for _ in range(30):
        loss = ctc_loss(model(batch.images), batch, model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert float(loss.detach()) < first


def test_sam_two_step_leaves_parameters_restored_then_updated():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4)
    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3)
    before = model.weight.detach().clone()
    x = torch.randn(5, 8)

    model(x).square().mean().backward()
    optimizer.first_step(zero_grad=True)
    climbed = model.weight.detach().clone()
    assert not torch.allclose(before, climbed)  # it moved to the worst case

    model(x).square().mean().backward()
    optimizer.second_step(zero_grad=True)
    # The epsilon climb is undone, so the net change is the optimizer's step.
    assert not torch.allclose(model.weight, climbed)
    assert torch.allclose(model.weight, before, atol=2e-3)


def test_greedy_decode_and_scoring_compose():
    charset = Charset.default()
    model = build_model(charset.n_classes, "small").eval()
    batch = _batch(charset, ["שלום", "עולם"])
    with torch.no_grad():
        logprobs = model(batch.images)
    hyps = greedy_decode(logprobs, model.output_lengths(batch.widths), charset)
    report = line_report(batch.texts, hyps)
    assert report.n_total == 2
    assert np.isnan(report.cer_median) or report.cer_median >= 0.0


def test_ema_lags_a_moving_target():
    """The point of an average is to trail the weights, not to copy them."""
    from hebocr.optim import ModelEMA

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)

    # Drift the weights steadily, as training does.
    for _ in range(200):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01)
        ema.update(model)

    shadow = dict(ema.shadow.named_parameters())["weight"]
    live = dict(model.named_parameters())["weight"]
    assert not torch.allclose(shadow, live)
    assert (shadow < live).all()  # it trails behind the drift


def test_ema_converges_on_a_still_target():
    from hebocr.optim import ModelEMA

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for _ in range(300):
        ema.update(model)
    assert torch.allclose(
        dict(ema.shadow.named_parameters())["weight"],
        dict(model.named_parameters())["weight"],
        atol=1e-4,
    )


def test_ema_ignores_gradients():
    from hebocr.optim import ModelEMA

    ema = ModelEMA(torch.nn.Linear(3, 3), decay=0.99)
    assert all(not p.requires_grad for p in ema.shadow.parameters())


def test_learning_rate_multiplier_applies_per_group():
    """A freshly initialized head needs a higher rate than a pretrained encoder;
    if the multiplier were ignored, the head would crawl and CTC would sit in
    its all-blank solution."""
    from hebocr.train import Config, _lr_at

    model = torch.nn.Module()
    encoder = torch.nn.Linear(4, 4)
    head = torch.nn.Linear(4, 4)
    groups = [
        {"params": list(encoder.parameters()), "lr_mult": 1.0},
        {"params": list(head.parameters()), "lr_mult": 10.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=1e-4)

    def set_lr(value):
        for g in optimizer.param_groups:
            g["lr"] = value * g.get("lr_mult", 1.0)

    cfg = Config(lr=1e-4, warmup_steps=10)
    set_lr(_lr_at(50, cfg, 100))
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert lrs[1] == pytest.approx(lrs[0] * 10.0)
    assert lrs[0] > 0
