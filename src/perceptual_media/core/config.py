"""YAML → plain dataclasses, strictly.

No Hydra/OmegaConf. A config is a YAML file whose keys must match a dataclass schema exactly:
unknown keys and missing required keys are errors that name the offending path
(``distortions[1].severty``). ``${name}`` references to top-level string keys are substituted
(used by ``configs/paths.yaml``).
"""

from __future__ import annotations

import dataclasses
import re
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised for unknown keys, missing required keys, or type mismatches."""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@dataclass
class CorpusConfig:
    """Which images to run over."""

    manifest: str = "data/corpus/manifest.csv"
    limit: int | None = None
    """Take only the first ``limit`` manifest rows (after class filtering)."""
    classes: list[str] | None = None
    """Restrict to these ``image_class`` values; ``None`` = all."""


@dataclass
class MarkerConfig:
    """Which embedding scheme to use."""

    name: str = "null"
    n_bits: int = 64
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistortionConfig:
    """One named distortion chain at one severity."""

    preset: str = "identity"
    severity: float = 0.0


@dataclass
class ExperimentConfig:
    """Top-level schema for ``pm-run``. One file = one run directory."""

    name: str
    seed: int = 0
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    distortions: list[DistortionConfig] = field(default_factory=lambda: [DistortionConfig()])
    strengths: list[float] = field(default_factory=lambda: [1.0])
    control: bool = True
    """Also run every image unmarked through the identical chain (false-positive accounting)."""
    output_dir: str = "outputs"
    device: str = "auto"
    """``auto`` = CUDA if available, else CPU."""


@dataclass
class PathsConfig:
    """Schema for ``configs/paths.yaml`` (everything outside the repo)."""

    external_root: str
    captures: str
    datasets: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading / dumping
# ---------------------------------------------------------------------------


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """``X | None`` / ``Optional[X]`` → ``(X, True)``; anything else → ``(tp, False)``."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _convert(value: Any, tp: Any, path: str) -> Any:
    tp, optional = _unwrap_optional(tp)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{path}: null is not allowed (expected {getattr(tp, '__name__', tp)})")

    if is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping for {tp.__name__}, got {type(value).__name__}")
        return _from_dict(tp, value, path)

    origin = typing.get_origin(tp)
    if origin is list:
        (item_tp,) = typing.get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_convert(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        key_tp, val_tp = typing.get_args(tp)
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {k: _convert(v, val_tp, f"{path}.{k}") for k, v in value.items()}
    if tp is Any:
        return value

    # Scalars. bool is checked first because bool is a subclass of int.
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if tp is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected float, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported schema type {tp!r}")


def _from_dict(cls: type[T], data: dict[str, Any], path: str) -> T:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        fpath = f"{path}.{f.name}" if path else f.name
        if f.name in data:
            kwargs[f.name] = _convert(data[f.name], hints[f.name], fpath)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{fpath}: required key is missing")
    return cls(**kwargs)


def _resolve_refs(data: dict[str, Any]) -> dict[str, Any]:
    """Substitute ``${key}`` in any string with the top-level string value of ``key``."""
    scalars = {k: v for k, v in data.items() if isinstance(v, str)}

    def sub(v: Any) -> Any:
        if isinstance(v, str):
            def repl(m: re.Match[str]) -> str:
                key = m.group(1)
                if key not in scalars:
                    raise ConfigError(f"unresolved reference ${{{key}}}")
                return scalars[key]
            return _REF.sub(repl, v)
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        return v

    return {k: sub(v) for k, v in data.items()}


def from_dict(schema: type[T], data: dict[str, Any]) -> T:
    """Build ``schema`` from a plain mapping, strictly (see module docstring)."""
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at top level, got {type(data).__name__}")
    return _from_dict(schema, _resolve_refs(data), "")


def load_config(path: str | Path, schema: type[T] = ExperimentConfig) -> T:  # type: ignore[assignment]
    """Load a YAML file into ``schema`` (default ``ExperimentConfig``)."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    try:
        return from_dict(schema, data)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def to_dict(cfg: Any) -> dict[str, Any]:
    """Dataclass → plain dict (recursively), suitable for YAML."""
    return dataclasses.asdict(cfg)


def dump_config(cfg: Any, path: str | Path) -> None:
    """Write ``cfg`` as YAML so that ``load_config(path, type(cfg)) == cfg``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False, allow_unicode=True)


def load_paths(path: str | Path = "configs/paths.yaml") -> PathsConfig:
    """Load the external-paths config."""
    return load_config(path, PathsConfig)
