"""Surfaces, not points (brief §6): the three figures every run gets, drawn from ``results.csv`` only.

1. ``ber_vs_severity.png`` — BER vs distortion severity, one line per image class, one panel per
   chain. The unmarked control set is a dashed reference at its own BER (≈ 0.5).
2. ``roc.png`` — presence-detection ROC per chain: marked vs control detector scores, AUC in
   the legend.
3. ``frontier.png`` — robustness–imperceptibility frontier: payload recovery rate vs LPIPS,
   one point per embed strength, per chain.

Image classes always map to the same colour (fixed order, never cycled) and marker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from perceptual_media.corpus.manifest import CLASSES  # noqa: E402
from perceptual_media.harness.results import read_results  # noqa: E402
from perceptual_media.metrics.decoding import roc  # noqa: E402

# Validated categorical palette (dataviz skill, light surface), fixed slot per class.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]
CLASS_STYLE = {c: (_PALETTE[i], _MARKERS[i]) for i, c in enumerate(CLASSES)}
_CONTROL = "#6b6b6b"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9a9a9a",
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "legend.frameon": False,
            "font.size": 9,
        }
    )


def _severity(params: str) -> float:
    try:
        return float(json.loads(params or "{}").get("severity", 0.0))
    except (ValueError, TypeError):
        return 0.0


def _title(df: pd.DataFrame, run_dir: Path, what: str) -> str:
    h = (run_dir / "git_hash.txt").read_text().strip() if (run_dir / "git_hash.txt").exists() else "?"
    return f"{what}\n{df['run_id'].iloc[0]} @ {h}"


def plot_ber_vs_severity(df: pd.DataFrame, run_dir: Path) -> Path:
    df = df.assign(severity=df["distortion_params"].map(_severity))
    chains = list(dict.fromkeys(df["distortion_chain"]))
    fig, axes = plt.subplots(1, len(chains), figsize=(4.2 * len(chains), 3.4), sharey=True, squeeze=False)
    for ax, chain in zip(axes[0], chains):
        g = df[df["distortion_chain"] == chain]
        m = g[g["marked"]]
        for cls in [c for c in CLASSES if c in set(m["image_class"])]:
            s = m[m["image_class"] == cls].groupby("severity")["ber"].agg(["mean", "std", "count"]).reset_index()
            color, marker = CLASS_STYLE[cls]
            ax.plot(s["severity"], s["mean"], marker=marker, color=color, label=cls)
            if (s["count"] > 1).any():
                sem = (s["std"].fillna(0) / np.sqrt(s["count"])).to_numpy()
                ax.fill_between(s["severity"], s["mean"] - sem, s["mean"] + sem, color=color, alpha=0.12, linewidth=0)
        c = g[~g["marked"]]
        if len(c):
            cs = c.groupby("severity")["ber"].mean().reset_index()
            ax.plot(cs["severity"], cs["ber"], linestyle="--", color=_CONTROL, label="unmarked control")
        ax.set_title(chain)
        ax.set_xlabel("severity")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02, 0.62)
    axes[0, 0].set_ylabel("BER (hard-sliced, before ECC)")
    axes[0, -1].legend(loc="lower right", fontsize=8)
    fig.suptitle(_title(df, run_dir, "BER vs severity"), fontsize=9)
    fig.tight_layout()
    out = run_dir / "ber_vs_severity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_roc(df: pd.DataFrame, run_dir: Path) -> Path:
    chains = list(dict.fromkeys(df["distortion_chain"]))
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color="#c8c8c8", linewidth=1, linestyle=":", label="chance")
    for i, chain in enumerate(chains):
        g = df[df["distortion_chain"] == chain]
        pos, neg = g.loc[g["marked"], "detector_score"].to_numpy(), g.loc[~g["marked"], "detector_score"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        fpr, tpr, auc = roc(pos, neg)
        ax.plot(fpr, tpr, color=_PALETTE[i % len(_PALETTE)], marker=_MARKERS[i % len(_MARKERS)], markevery=max(1, len(fpr) // 8), label=f"{chain} (AUC {auc:.2f}, n={len(pos)}+{len(neg)})")
    ax.set_xlabel("false-positive rate (unmarked control)")
    ax.set_ylabel("true-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(_title(df, run_dir, "Presence-detection ROC"), fontsize=8)
    fig.tight_layout()
    out = run_dir / "roc.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_frontier(df: pd.DataFrame, run_dir: Path) -> Path:
    m = df[df["marked"]]
    chains = list(dict.fromkeys(m["distortion_chain"]))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for i, chain in enumerate(chains):
        g = m[m["distortion_chain"] == chain].groupby("embed_strength").agg(lpips=("lpips", "mean"), rec=("payload_recovered", "mean"), n=("ber", "size")).reset_index().sort_values("embed_strength")
        ax.plot(g["lpips"], g["rec"], color=_PALETTE[i % len(_PALETTE)], marker=_MARKERS[i % len(_MARKERS)], label=chain)
        for _, r in g.iterrows():
            ax.annotate(f"s={r['embed_strength']:g}", (r["lpips"], r["rec"]), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#555555")
    ax.set_xlabel("LPIPS (marked vs original)  ← more invisible")
    ax.set_ylabel("payload recovery rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(_title(df, run_dir, "Robustness–imperceptibility frontier"), fontsize=8)
    fig.tight_layout()
    out = run_dir / "frontier.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_run(run_dir: str | Path) -> list[Path]:
    """Write all three figures into ``run_dir``; returns their paths."""
    _style()
    run_dir = Path(run_dir)
    df = read_results(run_dir)
    if df.empty:
        raise ValueError(f"{run_dir}: results.csv is empty")
    return [plot_ber_vs_severity(df, run_dir), plot_roc(df, run_dir), plot_frontier(df, run_dir)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pm-plot", description=__doc__)
    p.add_argument("run_dir", help="outputs/<run> directory containing results.csv")
    args = p.parse_args(argv)
    for path in plot_run(args.run_dir):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
