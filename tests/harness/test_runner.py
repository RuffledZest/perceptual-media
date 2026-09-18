import time
import warnings
from pathlib import Path

import pytest
import torch

from perceptual_media.core.config import DistortionConfig, ExperimentConfig, MarkerConfig, CorpusConfig
from perceptual_media.core.seed import make_generator
from perceptual_media.harness.cli import main, summarize
from perceptual_media.harness.results import read_results
from perceptual_media.harness.runner import (
    CorpusImage,
    builtin_smoke_corpus,
    load_corpus,
    run_experiment,
    trial_seed,
)

REPO = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, **kw) -> ExperimentConfig:  # type: ignore[no-untyped-def]
    base = dict(
        name="t",
        seed=0,
        corpus=CorpusConfig(manifest=str(tmp_path / "no-such-manifest.csv")),
        marker=MarkerConfig(name="null", n_bits=64),
        distortions=[DistortionConfig("identity", 0.0)],
        strengths=[1.0],
        control=True,
        output_dir=str(tmp_path / "out"),
        device="cpu",
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_builtin_corpus_classes() -> None:
    items = list(builtin_smoke_corpus(seed=0, size=16))
    assert [i.image_class for i in items] == ["flat", "flat", "gradient", "gradient", "textured", "textured"]
    for it in items:
        assert it.image.shape == (1, 3, 16, 16)


def test_load_corpus_filters_and_limits(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="built-in smoke corpus"):
        items = list(load_corpus(CorpusConfig(manifest=str(tmp_path / "x.csv"), classes=["flat", "textured"], limit=3)))
    assert [i.image_id for i in items] == ["flat00", "flat01", "textured00"]


def test_end_to_end_null_marker_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    t0 = time.perf_counter()
    with pytest.warns(UserWarning):
        run_dir = run_experiment(cfg)
    assert time.perf_counter() - t0 < 30
    df = read_results(run_dir)

    assert len(df) == 12  # 6 images × (marked + control)
    assert df["marked"].sum() == 6 and (~df["marked"]).sum() == 6
    assert set(df["image_class"]) == {"flat", "gradient", "textured"}
    assert 0.35 <= df["ber"].mean() <= 0.65
    assert (df["encode_ms"].dropna() >= 0).all() and (df["decode_ms"] > 0).all()
    assert df.loc[~df["marked"], "encode_ms"].isna().all()
    assert (df.loc[~df["marked"], "embed_strength"] == 0.0).all()
    assert (df["distortion_chain"] == "identity").all()
    assert (df["distortion_params"] == '{"identity": {}, "severity": 0.0}').all()
    # fidelity: marked rows populated (NullMarker → identical → PSNR cap, SSIM 1, LPIPS 0); control NaN
    m, c = df[df["marked"]], df[~df["marked"]]
    assert (m["psnr"] == 100.0).all() and (m["ssim"].sub(1).abs() < 1e-5).all() and (m["lpips"] <= 1e-6).all()
    assert c[["psnr", "ssim", "lpips"]].isna().all().all()
    assert (run_dir / "config.yaml").exists() and (run_dir / "git_hash.txt").exists()

    # marked and control rows of the same trial share the seed (identical distortion params)
    m = df[df["marked"]].set_index("image_id")["seed"]
    c = df[~df["marked"]].set_index("image_id")["seed"]
    assert (m == c).all()


def test_run_is_deterministic(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        a = read_results(run_experiment(_cfg(tmp_path)))
        b = read_results(run_experiment(_cfg(tmp_path)))
    for col in ("ber", "detector_score", "seed", "payload_recovered"):
        assert a[col].tolist() == b[col].tolist()


def test_control_off_and_multiple_strengths(tmp_path: Path) -> None:
    corpus = [CorpusImage("a", "flat", torch.rand(1, 3, 8, 8)), CorpusImage("b", "textured", torch.rand(1, 3, 8, 8))]
    cfg = _cfg(tmp_path, control=False, strengths=[0.5, 1.0, 2.0])
    df = read_results(run_experiment(cfg, corpus=corpus))
    assert len(df) == 2 * 3 and df["marked"].all()
    assert sorted(df["embed_strength"].unique()) == [0.5, 1.0, 2.0]


def test_trial_seed_stable_and_sensitive() -> None:
    s = trial_seed(0, "img", 1.0, "identity", 0.0)
    assert s == trial_seed(0, "img", 1.0, "identity", 0.0)
    assert s != trial_seed(1, "img", 1.0, "identity", 0.0)
    assert s != trial_seed(0, "img", 1.0, "identity", 0.5)
    assert 0 <= s < 2**32


def test_cli_smoke_config(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fallback warning only if data/corpus is not built
        rc = main([str(REPO / "configs" / "experiments" / "smoke.yaml"), "--output-dir", str(tmp_path), "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "identity" in out and "AUC=" in out and "control:" in out
    runs = list(tmp_path.glob("smoke-*"))
    assert len(runs) == 1
    assert len(read_results(runs[0])) >= 12 * 3


def test_unknown_preset_is_a_clear_error(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, distortions=[DistortionConfig("no_such_preset", 0.5)])
    with pytest.raises(KeyError, match="unknown distortion preset 'no_such_preset'"), pytest.warns(UserWarning):
        run_experiment(cfg)


def test_runner_with_ecc_reports_message_bits_and_recovery(tmp_path: Path) -> None:
    from perceptual_media.core.config import EccConfig

    g = make_generator(0)
    smooth = torch.nn.functional.interpolate(torch.rand(1, 3, 16, 16, generator=g), size=(256, 256), mode="bicubic", align_corners=False)
    corpus = [CorpusImage("a", "textured", (smooth + 0.05 * torch.randn(1, 3, 256, 256, generator=g)).clamp(0.05, 0.95))]
    cfg = _cfg(tmp_path, marker=MarkerConfig(name="classical_ss", n_bits=127), ecc=EccConfig("bch", 127, 64), control=True)
    df = read_results(run_experiment(cfg, corpus=corpus))
    m, c = df[df["marked"]], df[~df["marked"]]
    assert (df["n_payload_bits"] == 64).all()
    assert m["ber"].iloc[0] == 0.0 and bool(m["payload_recovered"].iloc[0])
    assert 0.3 < c["ber"].iloc[0] < 0.7 and not bool(c["payload_recovered"].iloc[0])


def test_runner_rejects_ecc_marker_mismatch(tmp_path: Path) -> None:
    from perceptual_media.core.config import EccConfig

    cfg = _cfg(tmp_path, marker=MarkerConfig(name="null", n_bits=64), ecc=EccConfig("bch", 127, 64))
    with pytest.raises(ValueError, match="must equal marker.n_bits"):
        run_experiment(cfg)
