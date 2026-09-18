from dataclasses import dataclass, field
from pathlib import Path

import pytest

from perceptual_media.core.config import (
    ConfigError,
    DistortionConfig,
    ExperimentConfig,
    PathsConfig,
    dump_config,
    from_dict,
    load_config,
    load_paths,
)

REPO = Path(__file__).resolve().parents[2]


def test_smoke_yaml_loads_into_nested_dataclasses() -> None:
    cfg = load_config(REPO / "configs" / "experiments" / "smoke.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "smoke"
    assert cfg.marker.name == "null" and cfg.marker.n_bits == 64
    assert cfg.corpus.limit == 8 and cfg.corpus.classes is None
    assert [d.preset for d in cfg.distortions] == ["identity", "print_camera", "screen_camera"]
    assert isinstance(cfg.distortions[1], DistortionConfig)
    assert cfg.distortions[1].severity == 0.5
    assert cfg.control is True


def test_defaults_fill_in() -> None:
    cfg = from_dict(ExperimentConfig, {"name": "x"})
    assert cfg.seed == 0
    assert cfg.strengths == [1.0]
    assert cfg.distortions == [DistortionConfig()]


def test_unknown_key_names_path() -> None:
    with pytest.raises(ConfigError, match=r"distortions\[1\]: unknown key\(s\) \['severty'\]"):
        from_dict(
            ExperimentConfig,
            {"name": "x", "distortions": [{"preset": "identity"}, {"severty": 1}]},
        )


def test_unknown_top_level_key() -> None:
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['nam'\]"):
        from_dict(ExperimentConfig, {"nam": "x"})


def test_missing_required_key_names_path() -> None:
    with pytest.raises(ConfigError, match=r"name: required key is missing"):
        from_dict(ExperimentConfig, {"seed": 1})


def test_type_mismatch_names_path() -> None:
    with pytest.raises(ConfigError, match=r"marker\.n_bits: expected int"):
        from_dict(ExperimentConfig, {"name": "x", "marker": {"n_bits": "64"}})
    with pytest.raises(ConfigError, match=r"control: expected bool"):
        from_dict(ExperimentConfig, {"name": "x", "control": 1})


def test_int_promotes_to_float_but_not_reverse() -> None:
    cfg = from_dict(ExperimentConfig, {"name": "x", "strengths": [1, 0.5]})
    assert cfg.strengths == [1.0, 0.5] and all(isinstance(s, float) for s in cfg.strengths)
    with pytest.raises(ConfigError, match="seed: expected int"):
        from_dict(ExperimentConfig, {"name": "x", "seed": 1.5})


def test_dump_then_load_is_equal(tmp_path: Path) -> None:
    cfg = load_config(REPO / "configs" / "experiments" / "smoke.yaml")
    out = tmp_path / "nested" / "cfg.yaml"
    dump_config(cfg, out)
    assert load_config(out) == cfg


def test_load_error_includes_filename(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bad.yaml"):
        load_config(p)


def test_reference_interpolation() -> None:
    cfg = from_dict(
        PathsConfig,
        {"external_root": "D:/root", "captures": "${external_root}/cap", "datasets": {"a": "${external_root}/a"}},
    )
    assert cfg.captures == "D:/root/cap" and cfg.datasets["a"] == "D:/root/a"
    with pytest.raises(ConfigError, match=r"unresolved reference \$\{nope\}"):
        from_dict(PathsConfig, {"external_root": "r", "captures": "${nope}"})


def test_repo_paths_yaml_loads() -> None:
    paths = load_paths(REPO / "configs" / "paths.yaml")
    assert paths.captures.startswith(paths.external_root)
    assert "landscape_pictures" in paths.datasets


def test_custom_schema_with_optional_nested() -> None:
    @dataclass
    class Inner:
        v: int = 1

    @dataclass
    class Outer:
        inner: Inner | None = None
        tags: list[str] = field(default_factory=list)

    assert from_dict(Outer, {}).inner is None
    assert from_dict(Outer, {"inner": {"v": 3}}).inner == Inner(3)
    with pytest.raises(ConfigError, match=r"tags\[0\]: expected str"):
        from_dict(Outer, {"tags": [1]})
