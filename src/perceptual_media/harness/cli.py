"""``pm-run <config.yaml>``: run an experiment and print a one-screen summary."""

from __future__ import annotations

import argparse
import sys

from perceptual_media.core.config import load_config
from perceptual_media.harness.results import read_results
from perceptual_media.harness.runner import run_experiment
from perceptual_media.metrics.decoding import auc


def summarize(run_dir: str) -> str:
    """Per-chain summary of a run: rows, mean BER (marked / control), recovery rate, AUC."""
    df = read_results(run_dir)
    lines = [f"run: {run_dir}", f"rows: {len(df)}  images: {df['image_id'].nunique()}  marker bits: {df['n_payload_bits'].iloc[0]}"]
    for chain, g in df.groupby("distortion_chain", sort=False):
        m, c = g[g["marked"]], g[~g["marked"]]
        line = f"  {chain:<16} marked: n={len(m)} BER={m['ber'].mean():.3f} recovered={m['payload_recovered'].mean():.2f}"
        if len(c):
            line += f" | control: n={len(c)} BER={c['ber'].mean():.3f}"
            line += f" | AUC={auc(m['detector_score'].to_numpy(), c['detector_score'].to_numpy()):.3f}"
        lines.append(line)
    lines.append(f"  encode {df['encode_ms'].mean():.2f} ms  decode {df['decode_ms'].mean():.2f} ms (mean)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pm-run", description=__doc__)
    p.add_argument("config", help="experiment YAML (see configs/experiments/)")
    p.add_argument("--output-dir", help="override cfg.output_dir")
    p.add_argument("--device", help="override cfg.device (auto|cpu|cuda)")
    p.add_argument("--seed", type=int, help="override cfg.seed")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    run_dir = run_experiment(cfg)
    print(summarize(str(run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
