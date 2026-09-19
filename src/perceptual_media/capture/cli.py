"""``pm-capture``: Week-4 capture tooling.

* ``pm-capture log <set>``      — build ``captures.csv`` (+ check overlays, rectified crops)
* ``pm-capture channel <set>``  — measure the real channel and match it to the simulator
* ``pm-capture sheet <spec>``   — build the display sheet (slides) for a decode round
* ``pm-capture decode <set> --sheet <name> [--controls <set>]`` — decode slide captures into a run dir
* ``pm-capture afc serve|report --sheet <name>`` — the 2AFC visibility study and its report
"""

from __future__ import annotations

import argparse
import json
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
    sh = sub.add_parser("sheet", help="build the display sheet (one slide per image x marker) from a spec YAML")
    sh.add_argument("spec", help="e.g. configs/capture_sheet.yaml")
    sh.add_argument("--corpus", default="data/corpus")
    dc = sub.add_parser("decode", help="locate, rectify and decode photos of display-sheet slides")
    dc.add_argument("set")
    dc.add_argument("--sheet", required=True, help="sheet name under paths.captures/sheets, or a directory")
    dc.add_argument("--controls", default=None, help="unmarked capture set (with captures.csv) for control rows")
    dc.add_argument("--config", default="configs/capture.yaml")
    dc.add_argument("--output-dir", default="outputs")
    af = sub.add_parser("afc", help="2AFC visibility study: serve the page, or report the responses")
    af.add_argument("action", choices=["serve", "report"])
    af.add_argument("--sheet", required=True)
    af.add_argument("--corpus", default="data/corpus")
    af.add_argument("--repeats", type=int, default=2)
    af.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "afc":
        from perceptual_media.capture.afc import Study, report, serve

        captures = Path(load_paths(args.paths).captures)
        sheet = Path(args.sheet) if Path(args.sheet).is_dir() else captures / "sheets" / args.sheet
        study = Study(sheet, Path(args.corpus), captures / "2afc" / sheet.name, repeats=args.repeats)
        if args.action == "serve":
            with open(sheet / "spec.json", encoding="utf-8") as fh:
                image_px = int(json.load(fh)["image_px"])
            serve(study, image_px, args.port)
            return 0
        if not study.responses.exists():
            print(f"no responses yet at {study.responses}")
            return 1
        report(study.responses)
        return 0

    if args.cmd == "sheet":
        from perceptual_media.capture.sheet import SheetSpec, build_sheet
        from perceptual_media.core.config import load_config

        spec = load_config(args.spec, SheetSpec)
        out = Path(load_paths(args.paths).captures) / "sheets" / spec.name
        df = build_sheet(spec, out, Path(args.corpus))
        cols = ["slide_id", "image_id", "marker", "strength", "psnr", "ssim", "lpips", "message_hex"]
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\n{len(df)} slides -> {out}")
        return 0

    folder = _resolve_set(args.set, args.paths)
    if args.cmd == "log":
        from perceptual_media.capture.log import log_folder, print_log

        df = log_folder(folder, load_capture_config(args.config))
        if df.empty:
            print(f"no captures found in {folder}")
            return 1
        print_log(df, folder)
        return 0

    if args.cmd == "decode":
        from perceptual_media.capture.ingest import decode_set

        sheet = Path(args.sheet)
        if not sheet.is_dir():
            sheet = Path(load_paths(args.paths).captures) / "sheets" / args.sheet
        controls = _resolve_set(args.controls, args.paths) if args.controls else None
        run_dir = decode_set(
            folder, sheet, load_capture_config(args.config), controls_dir=controls, output_dir=Path(args.output_dir)
        )
        df = pd.read_csv(run_dir / "results.csv")
        real = df[df["distortion_chain"] == "real_screen"]
        if not real.empty:
            cond = real["capture_conditions"].apply(lambda j: json.loads(j))
            real = real.assign(
                capture=cond.apply(lambda d: d["capture_id"]),
                marker=real["distortion_params"].apply(lambda j: json.loads(j)["marker"]),
            )
            with pd.option_context("display.width", 200, "display.max_rows", 500):
                print(
                    real[["capture", "marker", "ber", "payload_recovered", "detector_score"]].to_string(
                        index=False, float_format=lambda x: f"{x:.3f}"
                    )
                )
        n_rect = int((df["distortion_chain"] == "real_screen").sum())
        n_ctrl = int((df["distortion_chain"] == "real_screen_control").sum())
        print(f"\n{len(df)} rows ({n_rect} rectified, {n_ctrl} control) -> {run_dir}")
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
