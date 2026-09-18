"""The one row every experiment emits (brief §6), and the writer that persists it.

Layout of a run directory::

    outputs/<name>-<YYYYmmdd-HHMMSS>/
        config.yaml     resolved ExperimentConfig
        git_hash.txt    commit the run was made from ("+dirty" if uncommitted changes)
        results.csv     one ResultRow per trial, flushed after every row

Plots and reports read ``results.csv`` only — never in-memory state — so any figure can be
regenerated from the file.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from perceptual_media.core.config import dump_config


@dataclass
class ResultRow:
    """One trial: one image × one payload × one strength × one distortion chain × marked/control.

    Fidelity columns (``psnr``, ``ssim``, ``lpips``) are ``NaN`` for control rows (nothing was
    embedded, so there is no distortion to measure) and until Task 7 lands.
    ``capture_conditions`` is ``""`` for simulated channels and a JSON object for physical
    captures (phone, medium, distance_m, angle_deg, lighting, ...).
    """

    run_id: str
    seed: int
    image_id: str
    image_class: str
    marked: bool
    n_payload_bits: int
    embed_strength: float
    distortion_chain: str
    distortion_params: str  # JSON object
    capture_conditions: str  # JSON object or ""
    ber: float
    payload_recovered: bool
    detector_score: float
    psnr: float = math.nan
    ssim: float = math.nan
    lpips: float = math.nan
    encode_ms: float = math.nan
    decode_ms: float = math.nan
    capacity_bits: float = math.nan
    """``marker.capacity(img)`` for the *original* image, or NaN if the scheme has no estimate.
    Added after Week 2: the classical estimator's failure on flat images was invisible without it."""

    @staticmethod
    def columns() -> list[str]:
        return [f.name for f in fields(ResultRow)]


_DTYPES: dict[str, Any] = {
    "run_id": "string",
    "seed": "int64",
    "image_id": "string",
    "image_class": "string",
    "marked": "bool",
    "n_payload_bits": "int64",
    "embed_strength": "float64",
    "distortion_chain": "string",
    "distortion_params": "string",
    "capture_conditions": "string",
    "ber": "float64",
    "payload_recovered": "bool",
    "detector_score": "float64",
    "psnr": "float64",
    "ssim": "float64",
    "lpips": "float64",
    "encode_ms": "float64",
    "decode_ms": "float64",
    "capacity_bits": "float64",
}


def to_json(params: dict[str, Any] | None) -> str:
    """Serialise a params dict deterministically (sorted keys) for the CSV."""
    return json.dumps(params or {}, sort_keys=True, default=_json_default)


def _json_default(o: Any) -> Any:
    # torch/numpy scalars and tensors → python
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def git_hash(repo: str | Path | None = None) -> str:
    """Short commit hash of ``repo`` (default: cwd), with ``+dirty`` if the tree has changes."""
    try:
        cwd = str(repo) if repo else None
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_dir(output_dir: str | Path, name: str, when: datetime | None = None) -> Path:
    """``<output_dir>/<name>-<YYYYmmdd-HHMMSS>``, created. Suffixes ``-2``, ``-3``… on collision."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = Path(output_dir) / f"{name}-{stamp}"
    run_dir, k = base, 1
    while run_dir.exists():
        k += 1
        run_dir = base.with_name(f"{base.name}-{k}")
    run_dir.mkdir(parents=True)
    return run_dir


class ResultWriter:
    """Append ``ResultRow``s to ``<run_dir>/results.csv``; flush after every row.

    Use as a context manager. On open, writes ``config.yaml`` (if a config is given) and
    ``git_hash.txt`` so the run is attributable even if it crashes half-way.
    """

    def __init__(self, run_dir: str | Path, config: Any | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "results.csv"
        self._config = config
        self._fh: Any = None
        self._writer: csv.DictWriter | None = None
        self.n_rows = 0

    def __enter__(self) -> ResultWriter:
        if self._config is not None:
            dump_config(self._config, self.run_dir / "config.yaml")
        (self.run_dir / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=ResultRow.columns())
        if new:
            self._writer.writeheader()
            self._fh.flush()
        return self

    def write(self, row: ResultRow) -> None:
        if self._writer is None or self._fh is None:
            raise RuntimeError("ResultWriter must be used as a context manager")
        d = asdict(row)
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                d[k] = ""  # pandas reads "" in a float column as NaN
        self._writer.writerow(d)
        self._fh.flush()
        self.n_rows += 1

    def __exit__(self, *exc: Any) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


def write_summary(run_dir: str | Path) -> Path:
    """Per-chain x severity summary of a run as ``summary.json`` (file-based comparison across runs).

    Keys per entry: ``n_marked``, ``n_control``, ``ber``, ``ber_control``, ``recovery``,
    ``recovery_control`` (false positives), ``auc``, ``tpr_at_1pct_fpr`` (NaN without controls),
    ``psnr``, ``ssim``, ``lpips``, ``encode_ms``, ``decode_ms``.
    """
    from perceptual_media.metrics.decoding import (  # avoid import cycle at module load
        roc,
        tpr_at_fpr,
    )

    run_dir = Path(run_dir)
    df = read_results(run_dir)
    df = df.assign(severity=df["distortion_params"].map(lambda p: json.loads(p or "{}").get("severity", 0.0)))
    out: dict[str, Any] = {"run_id": str(df["run_id"].iloc[0]) if len(df) else run_dir.name, "rows": int(len(df)), "chains": []}
    for (chain, sev), g in df.groupby(["distortion_chain", "severity"], sort=False):
        m, c = g[g["marked"]], g[~g["marked"]]
        entry: dict[str, Any] = {
            "chain": str(chain), "severity": float(sev),
            "n_marked": int(len(m)), "n_control": int(len(c)),
            "ber": float(m["ber"].mean()) if len(m) else math.nan,
            "recovery": float(m["payload_recovered"].mean()) if len(m) else math.nan,
            "ber_control": float(c["ber"].mean()) if len(c) else math.nan,
            "recovery_control": float(c["payload_recovered"].mean()) if len(c) else math.nan,
            "psnr": float(m["psnr"].mean()), "ssim": float(m["ssim"].mean()), "lpips": float(m["lpips"].mean()),
            "encode_ms": float(m["encode_ms"].mean()), "decode_ms": float(g["decode_ms"].mean()),
            "auc": math.nan, "tpr_at_1pct_fpr": math.nan,
        }
        if len(m) and len(c):
            pos, neg = m["detector_score"].to_numpy(), c["detector_score"].to_numpy()
            entry["auc"] = roc(pos, neg)[2]
            entry["tpr_at_1pct_fpr"] = tpr_at_fpr(pos, neg, 0.01)
        out["chains"].append(entry)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=True), encoding="utf-8")
    return path


def read_results(run_dir_or_csv: str | Path) -> pd.DataFrame:
    """Load ``results.csv`` with the ``ResultRow`` dtypes enforced."""
    p = Path(run_dir_or_csv)
    if p.is_dir():
        p = p / "results.csv"
    df = pd.read_csv(p, dtype={k: v for k, v in _DTYPES.items() if v != "bool"}, keep_default_na=True)
    for col in ("marked", "payload_recovered"):
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False}).astype("bool")
    for col in ("distortion_params", "capture_conditions"):
        df[col] = df[col].fillna("").astype("string")
    return df[ResultRow.columns()]
