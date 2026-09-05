"""Guards on the repository itself, not the model.

These exist because `.gitignore` contained a bare `data/`, which matches
`hebocr/data/` as well as the top-level data directory. The entire data package
-- augmentation, datasets, glyph composition -- was silently never committed,
and the published repository could not even be imported. Tests passed the whole
time, because they ran against the working tree.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def test_every_package_module_is_committed():
    """A module that exists only in the working tree is a module nobody else has."""
    tracked = tracked_files()
    on_disk = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "hebocr").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, f"untracked package modules: {missing}"


def test_every_test_module_is_committed():
    tracked = tracked_files()
    on_disk = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "tests").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert not sorted(on_disk - tracked)


def test_scripts_are_committed():
    tracked = tracked_files()
    on_disk = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "scripts").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert not sorted(on_disk - tracked)


def test_the_package_imports_from_a_clean_checkout():
    """Import every module, so a missing dependency between them is caught."""
    import importlib

    for name in (
        "hebocr.normalize", "hebocr.metrics", "hebocr.charset", "hebocr.decode",
        "hebocr.lm", "hebocr.optim", "hebocr.recognize", "hebocr.train",
        "hebocr.evaluate", "hebocr.models.htr_vt", "hebocr.page.segment",
        "hebocr.data.transforms", "hebocr.data.synth", "hebocr.data.glyphs",
        "hebocr.data.benchmark",
    ):
        importlib.import_module(name)


def test_exported_checkpoint_loads_and_reads(tmp_path):
    """A release file must be enough on its own: weights, charset, config."""
    import subprocess
    import sys

    import numpy as np
    import torch
    from PIL import Image

    from hebocr.charset import Charset
    from hebocr.models.htr_vt import build_model
    from hebocr.recognize import Recognizer

    charset = Charset.default()
    model = build_model(charset.n_classes, "small", mask_ratio=0.0)
    full = tmp_path / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "raw_model": model.state_dict(),
            "optimizer": {"bloat": torch.zeros(1_000_00)},
            "charset": charset.chars,
            "config": {"size": "small", "epochs": 18},
            "epoch": 3,
            "used_ema": True,
        },
        full,
    )

    released = tmp_path / "release.pt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_model.py"), str(full), "--out", str(released)],
        check=True, capture_output=True, cwd=ROOT,
    )
    assert released.stat().st_size < full.stat().st_size

    recognizer = Recognizer(released, device="cpu")
    image = Image.fromarray(np.full((64, 240), 200, np.uint8), mode="L")
    assert isinstance(recognizer.read([image])[0], str)
