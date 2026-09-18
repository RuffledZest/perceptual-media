from pathlib import Path

import pytest
import torch

from perceptual_media.core.config import CorpusConfig, DistortionConfig, ExperimentConfig, MarkerConfig
from perceptual_media.harness.plots import CLASS_STYLE, main, plot_run
from perceptual_media.harness.runner import CorpusImage, run_experiment


def _run(tmp_path: Path) -> Path:
    corpus = [
        CorpusImage("f", "flat", torch.full((1, 3, 32, 32), 0.5)),
        CorpusImage("t", "textured", torch.rand(1, 3, 32, 32)),
    ]
    cfg = ExperimentConfig(
        name="p", seed=0, corpus=CorpusConfig(manifest=str(tmp_path / "none.csv")),
        marker=MarkerConfig(name="null", n_bits=16),
        distortions=[DistortionConfig("identity", 0.0), DistortionConfig("print_camera", 0.5), DistortionConfig("print_camera", 1.0)],
        strengths=[0.5, 1.0], control=True, output_dir=str(tmp_path / "out"), device="cpu",
    )
    return run_experiment(cfg, corpus=corpus)


def test_plot_run_writes_three_nonempty_figures(tmp_path: Path) -> None:
    run = _run(tmp_path)
    paths = plot_run(run)
    assert [p.name for p in paths] == ["ber_vs_severity.png", "roc.png", "frontier.png"]
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_and_fixed_class_colours(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path)
    assert main([str(run)]) == 0
    assert "roc.png" in capsys.readouterr().out
    # colour follows the class, not its position in a given run
    assert CLASS_STYLE["flat"][0] != CLASS_STYLE["textured"][0]
    assert len({v[0] for v in CLASS_STYLE.values()}) == len(CLASS_STYLE)


def test_empty_results_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text("run_id,seed\n")
    with pytest.raises(Exception):
        plot_run(tmp_path)
