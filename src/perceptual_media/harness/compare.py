"""``pm-compare <run_a> <run_b> [...]``: overlay runs on the same axes from their ``summary.json``.

Writes ``compare_ber.png`` (raw BER vs severity, one panel per chain, one line per run) and
``compare_recovery.png`` (payload recovery vs severity) into the first run's directory, and
prints a per-chain × severity table. Runs are labelled by their marker name from ``config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _label(run_dir: Path) -> str:
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        d = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        m = d.get("marker", {})
        return f"{m.get('name', '?')} ({run_dir.name})"
    return run_dir.name


def load_summary(run_dir: str | Path) -> dict:
    """``summary.json`` of a run; regenerated from ``results.csv`` if the run predates it."""
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        from perceptual_media.harness.results import write_summary

        write_summary(run_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def compare_runs(run_dirs: list[str | Path], metric: str, out: Path, ylabel: str, ylim: tuple[float, float]) -> Path:
    runs = [(Path(r), load_summary(r)) for r in run_dirs]
    chains = list(dict.fromkeys(e["chain"] for _, s in runs for e in s["chains"]))
    fig, axes = plt.subplots(1, len(chains), figsize=(3.6 * len(chains), 3.4), sharey=True, squeeze=False)
    for ax, chain in zip(axes[0], chains, strict=True):
        for i, (rd, s) in enumerate(runs):
            pts = sorted((e["severity"], e[metric]) for e in s["chains"] if e["chain"] == chain)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=_MARKERS[i % 6], color=_PALETTE[i % 6], label=_label(rd))
        ax.set_title(chain, fontsize=9)
        ax.set_xlabel("severity")
        ax.set_ylim(*ylim)
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def table(run_dirs: list[str | Path]) -> str:
    runs = [(Path(r), load_summary(r)) for r in run_dirs]
    keys = sorted({(e["chain"], e["severity"]) for _, s in runs for e in s["chains"]})
    head = f"{'chain':18s} {'sev':>5s} | " + " | ".join(f"{_label(rd)[:28]:>28s}" for rd, _ in runs)
    lines = [head, "-" * len(head), f"{'':18s} {'':>5s} | " + " | ".join(f"{'BER   rec   AUC   LPIPS':>28s}" for _ in runs)]
    for chain, sev in keys:
        cells = []
        for _, s in runs:
            e = next((e for e in s["chains"] if e["chain"] == chain and e["severity"] == sev), None)
            cells.append(f"{e['ber']:.3f} {e['recovery']:.2f} {e['auc']:.2f} {e['lpips']:.4f}".rjust(28) if e else " " * 28)
        lines.append(f"{chain:18s} {sev:5.2f} | " + " | ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pm-compare", description=__doc__)
    p.add_argument("runs", nargs="+", help="two or more outputs/<run> directories with summary.json")
    args = p.parse_args(argv)
    out_dir = Path(args.runs[0])
    print(table(args.runs))
    print(compare_runs(args.runs, "ber", out_dir / "compare_ber.png", "raw BER (before ECC)", (-0.02, 0.62)))
    print(compare_runs(args.runs, "recovery", out_dir / "compare_recovery.png", "payload recovery rate", (-0.02, 1.02)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
