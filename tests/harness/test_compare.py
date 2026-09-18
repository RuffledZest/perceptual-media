from pathlib import Path

import pytest
import torch

from perceptual_media.core.config import (
    CorpusConfig,
    DistortionConfig,
    ExperimentConfig,
    MarkerConfig,
)
from perceptual_media.harness.compare import compare_runs, load_summary, main, table
from perceptual_media.harness.runner import CorpusImage, run_experiment


def _run(tmp_path: Path, name: str, n_bits: int) -> Path:
    corpus = [CorpusImage("a", "flat", torch.full((1, 3, 32, 32), 0.5)), CorpusImage("b", "textured", torch.rand(1, 3, 32, 32))]
    cfg = ExperimentConfig(
        name=name, seed=0, corpus=CorpusConfig(manifest=str(tmp_path / "none.csv")),
        marker=MarkerConfig(name="null", n_bits=n_bits),
        distortions=[DistortionConfig("identity", 0.0), DistortionConfig("print_camera", 0.5), DistortionConfig("print_camera", 1.0)],
        strengths=[1.0], control=True, output_dir=str(tmp_path / "out"), device="cpu",
    )
    return run_experiment(cfg, corpus=corpus)


def test_compare_two_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _run(tmp_path, "a", 32), _run(tmp_path, "b", 64)
    assert load_summary(a)["rows"] == 12
    t = table([a, b])
    assert "print_camera" in t and "identity" in t
    out = compare_runs([a, b], "ber", a / "compare_ber.png", "BER", (0, 0.6))
    assert out.exists() and out.stat().st_size > 1000
    assert main([str(a), str(b)]) == 0
    assert "compare_recovery.png" in capsys.readouterr().out
