import pytest
import torch

from hebocr.charset import Charset
from hebocr.models.htr_vt import build_model


@pytest.fixture(scope="module")
def charset():
    return Charset.default()


def test_output_length_matches_the_declared_stride(charset):
    """CTC needs the time-step count to be exactly what `output_lengths` says."""
    model = build_model(charset.n_classes, "small").eval()
    for width in (128, 256, 512, 1024, 2048):
        with torch.no_grad():
            out = model(torch.randn(1, 1, 64, width))
        assert out.shape[1] == width // 4
        assert int(model.output_lengths(torch.tensor([width]))[0]) == out.shape[1]


def test_emits_log_probabilities(charset):
    model = build_model(charset.n_classes, "small").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 64, 256))
    assert out.shape[-1] == charset.n_classes
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 64), atol=1e-4)
    assert (out <= 0).all()


def test_handles_the_widest_benchmark_line(charset):
    """The longest gold line needs ~2433 px at a 64 px height."""
    model = build_model(charset.n_classes, "small").eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 2560))
    assert out.shape[1] == 640


def test_span_mask_is_training_only(charset):
    """A regularizer that fired at inference would make results irreproducible."""
    model = build_model(charset.n_classes, "small", mask_ratio=0.9, dropout=0.0)
    x = torch.randn(1, 1, 64, 256)
    model.eval()
    with torch.no_grad():
        first, second = model(x), model(x)
    assert torch.allclose(first, second)


def test_span_mask_perturbs_during_training(charset):
    model = build_model(charset.n_classes, "small", mask_ratio=0.9, dropout=0.0)
    model.train()
    x = torch.randn(1, 1, 64, 256)
    assert not torch.allclose(model(x), model(x))


def test_presets_are_ordered_by_capacity(charset):
    sizes = [sum(p.numel() for p in build_model(charset.n_classes, s).parameters())
             for s in ("small", "base", "large")]
    assert sizes == sorted(sizes)


def test_unknown_preset_is_rejected(charset):
    with pytest.raises(ValueError):
        build_model(charset.n_classes, "enormous")


def test_trocr_ctc_shapes_and_step_count(charset):
    """The pretrained-encoder variant must honour the same CTC contract."""
    import torch as t

    from hebocr.models.trocr_ctc import build_trocr_ctc
    from hebocr.recognize import _bare_vit

    model = build_trocr_ctc(
        charset.n_classes, grad_checkpointing=False, encoder=_bare_vit({})
    ).eval()
    for width in (512, 1024, 2432):
        with t.no_grad():
            out = model(t.rand(1, 1, 64, width))
        assert out.shape[1] == int(model.output_lengths(t.tensor([width]))[0])
        assert out.shape[2] == charset.n_classes
    # The longest benchmark line is 111 characters; it must fit comfortably.
    assert int(model.output_lengths(t.tensor([2432]))[0]) > 111 * 2


def test_trocr_ctc_freezing_actually_freezes(charset):
    from hebocr.models.trocr_ctc import build_trocr_ctc
    from hebocr.recognize import _bare_vit

    free = build_trocr_ctc(charset.n_classes, grad_checkpointing=False, encoder=_bare_vit({}))
    frozen = build_trocr_ctc(
        charset.n_classes, freeze_layers=6, grad_checkpointing=False, encoder=_bare_vit({})
    )
    n_free = sum(p.numel() for p in free.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    # The attribute holding the blocks moved between transformers versions; if
    # the lookup silently failed, nothing would be frozen and these would match.
    assert n_frozen < n_free * 0.75


def test_trocr_ctc_checkpoint_roundtrip(charset, tmp_path):
    """A saved trocr checkpoint must rebuild without downloading anything."""
    import torch as t

    from hebocr.models.trocr_ctc import build_trocr_ctc
    from hebocr.recognize import Recognizer, _bare_vit

    model = build_trocr_ctc(charset.n_classes, grad_checkpointing=False, encoder=_bare_vit({}))
    path = tmp_path / "trocr.pt"
    t.save(
        {"model": model.state_dict(), "charset": charset.chars,
         "config": {"arch": "trocr"}, "epoch": 0},
        path,
    )

    import numpy as np
    from PIL import Image

    recognizer = Recognizer(path, device="cpu")
    image = Image.fromarray(np.full((64, 320), 200, np.uint8), mode="L")
    assert isinstance(recognizer.read([image])[0], str)
