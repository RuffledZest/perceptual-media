"""``pm-capture channel <set>``: measure the real channel per capture and match it to the
simulator (``capture.channel``). Writes into the set folder:

* ``channel.csv`` — one row per located capture: conditions (nominal + measured), the
  ``ChannelStats`` of its rectified crop, and ``matched_severity`` / ``match_distance``;
* ``channel_sim.csv`` — the simulator's ``ChannelStats`` for every reference image at every
  severity of the grid (mean over seeds);
* ``channel_match.csv`` — matched severity per nominal condition (mean / min / max over images);
* ``channel_fit.png`` — the four matched statistics vs severity: simulator band (min–max over
  images) with each real capture drawn at its matched severity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from perceptual_media.capture.channel import MATCH_KEYS, ChannelStats, channel_stats, match_severity, sim_channel_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEVERITIES = tuple(round(0.1 * i, 1) for i in range(11))


def _reference(image_id: str, corpus: Path, manifest: pd.DataFrame) -> np.ndarray:
    return np.asarray(Image.open(corpus / manifest.loc[image_id, "path"]).convert("RGB"))


def calibrate(
    set_dir: Path,
    corpus: Path = Path("data/corpus"),
    severities: tuple[float, ...] = SEVERITIES,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """Run the whole calibration for one capture set; returns the per-capture table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(corpus / "manifest.csv").set_index("image_id")
    caps = pd.read_csv(set_dir / "captures.csv")
    caps = caps[caps["locate_ok"]].reset_index(drop=True)
    image_ids = sorted(caps["image_id"].unique())

    # Simulator grid: every reference × severity, mean over seeds.
    sim_rows = []
    grids: dict[str, list[tuple[float, dict[str, float]]]] = {}
    for iid in image_ids:
        ref = _reference(iid, corpus, manifest)
        grids[iid] = []
        for sev in severities:
            stats = sim_channel_stats(ref, sev, seeds=seeds, device=device)
            grids[iid].append((sev, stats))
            sim_rows.append({"image_id": iid, "severity": sev, **stats})
        print(f"sim grid: {iid} done", flush=True)
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(set_dir / "channel_sim.csv", index=False)

    # Real captures.
    rows = []
    for _, c in caps.iterrows():
        ref = _reference(c["image_id"], corpus, manifest)
        cap = np.asarray(Image.open(set_dir / "rectified" / f"{c['capture_id']}.png").convert("RGB"))
        stats = channel_stats(ref, cap, device)
        sev, dist = match_severity(stats.__dict__, grids[c["image_id"]])
        keep = ("capture_id", "image_id", "image_class", "medium", "distance_m", "angle_deg", "lighting",
                "measured_distance_m", "measured_angle_deg", "px_per_image_px")
        rows.append({**{k: c[k] for k in keep}, **stats.__dict__, "matched_severity": sev, "match_distance": dist})
    real = pd.DataFrame(rows)
    real.to_csv(set_dir / "channel.csv", index=False)

    real["condition"] = real.apply(lambda r: f"d{r.distance_m:g} a{r.angle_deg:g}", axis=1)
    summary = real.groupby("condition")["matched_severity"].agg(["mean", "min", "max", "count"]).reset_index()
    summary.to_csv(set_dir / "channel_match.csv", index=False)
    plot_fit(real, sim, set_dir / "channel_fit.png")
    return real


def plot_fit(real: pd.DataFrame, sim: pd.DataFrame, out: Path) -> None:
    """Simulator band vs severity for each matched statistic, with real captures at their match."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    conds = sorted(real["condition"].unique())
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(conds)))
    for ax, key in zip(axes.ravel(), MATCH_KEYS, strict=True):
        g = sim.groupby("severity")[key]
        sev = np.array(sorted(sim["severity"].unique()))
        ax.fill_between(
            sev, g.min().values, g.max().values, alpha=0.2, color="grey", label="simulator (min–max over images)"
        )
        ax.plot(sev, g.mean().values, color="k", lw=1.5, label="simulator mean")
        for cond, col in zip(conds, colours, strict=True):
            sub = real[real["condition"] == cond].dropna(subset=[key])
            ax.scatter(sub["matched_severity"], sub[key], color=col, s=28, zorder=3, label=f"real {cond}")
        ax.set_xlabel("simulator severity")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Real screen→camera channel vs simulator (screen_channel preset); each point at its matched severity")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


__all__ = ["ChannelStats", "calibrate", "plot_fit"]
