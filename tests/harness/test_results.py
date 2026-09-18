import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from perceptual_media.core.config import ExperimentConfig, load_config
from perceptual_media.harness.results import (
    ResultRow,
    ResultWriter,
    git_hash,
    make_run_dir,
    read_results,
    to_json,
)

BRIEF_S6_FIELDS = [
    "run_id", "seed", "image_id", "image_class", "marked", "n_payload_bits", "embed_strength",
    "distortion_chain", "distortion_params", "capture_conditions", "ber", "payload_recovered",
    "detector_score", "psnr", "ssim", "lpips", "encode_ms", "decode_ms",
]


def _row(i: int, marked: bool = True) -> ResultRow:
    return ResultRow(
        run_id="t", seed=0, image_id=f"img{i:03d}", image_class="flat", marked=marked,
        n_payload_bits=64, embed_strength=1.0, distortion_chain="identity",
        distortion_params=to_json({"severity": 0.0}), capture_conditions="",
        ber=i / 100, payload_recovered=(i % 2 == 0), detector_score=float(i),
        psnr=40.0 if marked else math.nan, ssim=0.99 if marked else math.nan,
        lpips=0.01 if marked else math.nan, encode_ms=1.5, decode_ms=2.5,
    )


def test_schema_matches_brief_section_6_exactly() -> None:
    assert ResultRow.columns() == BRIEF_S6_FIELDS


def test_write_100_rows_and_read_back_with_dtypes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with ResultWriter(run, config=ExperimentConfig(name="t")) as w:
        for i in range(100):
            w.write(_row(i, marked=(i % 3 != 0)))
        assert w.n_rows == 100
    assert (run / "config.yaml").exists()
    assert (run / "git_hash.txt").read_text().strip() != ""
    assert load_config(run / "config.yaml").name == "t"

    df = read_results(run)
    assert len(df) == 100
    assert list(df.columns) == BRIEF_S6_FIELDS
    assert df["marked"].dtype == bool and df["payload_recovered"].dtype == bool
    assert df["ber"].dtype == np.float64 and df["seed"].dtype == np.int64
    assert df["marked"].sum() == sum(1 for i in range(100) if i % 3 != 0)
    assert df["payload_recovered"].sum() == 50
    # NaN fidelity on control rows, present on marked rows
    assert df.loc[~df["marked"], "psnr"].isna().all()
    assert (df.loc[df["marked"], "psnr"] == 40.0).all()
    assert (df["capture_conditions"] == "").all()
    assert df["distortion_params"].iloc[0] == '{"severity": 0.0}'


def test_rows_are_flushed_immediately(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with ResultWriter(run) as w:
        w.write(_row(0))
        # read while still open: header + 1 row must already be on disk
        lines = (run / "results.csv").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2


def test_append_to_existing_csv_does_not_duplicate_header(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with ResultWriter(run) as w:
        w.write(_row(0))
    with ResultWriter(run) as w:
        w.write(_row(1))
    df = read_results(run)
    assert len(df) == 2 and list(df["image_id"]) == ["img000", "img001"]


def test_write_outside_context_raises(tmp_path: Path) -> None:
    w = ResultWriter(tmp_path / "run")
    with pytest.raises(RuntimeError):
        w.write(_row(0))


def test_make_run_dir_is_unique(tmp_path: Path) -> None:
    when = datetime(2026, 9, 18, 12, 0, 0)
    a = make_run_dir(tmp_path, "smoke", when)
    b = make_run_dir(tmp_path, "smoke", when)
    assert a.name == "smoke-20260918-120000" and b.name == "smoke-20260918-120000-2"
    assert a.is_dir() and b.is_dir()


def test_to_json_is_deterministic_and_handles_tensors() -> None:
    a = to_json({"b": 1, "a": torch.tensor(0.5), "c": np.float32(2.0), "d": torch.tensor([1, 2])})
    assert a == '{"a": 0.5, "b": 1, "c": 2.0, "d": [1, 2]}'
    assert to_json(None) == "{}"


def test_git_hash_from_repo() -> None:
    h = git_hash(Path(__file__).resolve().parents[2])
    assert h == "unknown" or len(h.split("+")[0]) >= 7
