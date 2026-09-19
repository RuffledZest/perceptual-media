"""``pm-capture``: Week-4 capture tooling.

* ``pm-capture log <set>``      — build ``captures.csv`` (+ check overlays, rectified crops)
* ``pm-capture channel <set>``  — measure the real channel and match it to the simulator
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from perceptual_media.core.config import load_capture_config, load_paths


def _resolve_set(name: str, paths_cfg: str) -> Path:
    folder = Path(name)
    if not folder.is_dir():
        folder = Path(load_paths(paths_cfg).captures) / name
    if not folder.is_dir():
        raise SystemExit(f"no such capture set: {folder}")
    return folder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pm-capture", description=__doc__.split("\n")[0])
    ap.add_argument("--paths", default="configs/paths.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="build captures.csv for a folder of capture photos")
    lg.add_argument("set", help="capture set: a folder name under paths.captures, or a directory")
    lg.add_argument("--config", default="configs/capture.yaml")
    ch = sub.add_parser("channel", help="measure the real channel per capture and match it to the simulator")
    ch.add_argument("set")
    ch.add_argument("--corpus", default="data/corpus")
    args = ap.parse_args(argv)

    folder = _resolve_set(args.set, args.paths)
    if args.cmd == "log":
        from perceptual_media.capture.log import log_folder, print_log

        df = log_folder(folder, load_capture_config(args.config))
        if df.empty:
            print(f"no captures found in {folder}")
            return 1
        print_log(df, folder)
        return 0

    from perceptual_media.capture.calibrate import calibrate

    real = calibrate(folder, Path(args.corpus))
    cols = ["capture_id", "blur_sigma", "noise_std", "moire_amp", "psnr_cc", "matched_severity", "match_distance"]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(real[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print()
        print(pd.read_csv(folder / "channel_match.csv").to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n-> {folder / 'channel.csv'}, channel_sim.csv, channel_match.csv, channel_fit.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
