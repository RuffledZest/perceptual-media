# Capture protocol (Week 4)

How phone photos of displayed / printed images are taken, named, logged and turned into
measurements. Two capture sets exist:

| set | what is shown | purpose |
|---|---|---|
| **channel calibration** | unmarked corpus images | measure the real screen→camera channel and fit the simulator to it |
| **decode** (Task 24/25) | marked + control images from the display sheet | the Week-4 robustness surface and the GO/NO-GO |

Everything lives under `configs/paths.yaml → captures` (external, irreplaceable — keep it synced),
one folder per set, e.g. `captures/screen_calib_20260919/`.

## Rig

Recorded once in `configs/capture.yaml`, not per photo:

- **display** — name, native resolution, pixel pitch (mm). The viewer shows the image at **1:1
  pixel zoom, centred, on a dark uniform background** (Windows Photos full-screen, dark theme).
  No scaling: the display's own upsampling filter would otherwise be part of the channel.
- **camera** — phone and its 35 mm-equivalent focal length (phone EXIF omits it; needed for the
  distance estimate). Main camera only, no digital zoom, no flash, default JPEG output.
- `image_px` — side of the displayed image (512 for the corpus).

Session notes go in `docs/worklog.md`: room lighting, whether the panel brightness was changed,
anything unusual.

## Shooting

For each image and each condition, one photo. Conditions used for the calibration set:

```
distance {0.5 m, 1 m} × angle {0°, 30°, 45°}, reduced to
0.5 m × {0°, 30°, 45°}  +  1 m × 0°            (4 per image, 24 total for 6 images)
```

- Distance and angle are **nominal** — by eye is fine. The logger measures the real values from
  the geometry (PnP on the located square) and the writeup uses those.
- **Tap the image on the phone screen before shooting** so the camera exposes for the image,
  not the dark room. Do not otherwise fight the phone: auto-exposure, auto-WB, HDR, moiré,
  reflections and clipping are the channel. (Calibration set note: the radial vignette's
  centre still clips; brightness was deliberately *not* lowered, to keep the set uniform.)
- Framing does not matter; the image only needs to be fully inside the frame and unobstructed.
  Background clutter (desk, cables, other monitors, ceiling lights) is fine — the locator uses
  the panel itself as the fiducial and the marked-image sheet will carry its own.
- Avoid ceiling-light reflections *on the image*; reflections elsewhere on the panel are fine.

## Naming

```
<image_id>__<medium>__d<distance_m>__a<angle_deg>__<lighting>[__m<0|1>].jpg
photo_003__screen__d0.5__a30__room.jpg        control / unmarked (default)
photo_003__print__d1__a0__led__m1.jpg         marked
```

`medium` ∈ {`screen`, `print`}; `lighting` is a free token (`room`, `led`, `daylight`, …).
A doubled extension (`.jpg.jpg`) from export tools is tolerated.

## Logging

```
uv run pm-capture log <set>          # <set> = folder name under paths.captures, or a path
```

Writes `<set>/captures.csv` (regenerated from the files every time; nothing is typed by hand):

| column group | fields |
|---|---|
| nominal (filename) | `image_id, image_class, medium, marked, distance_m, angle_deg, lighting` |
| camera (EXIF) | `camera, iso, exposure_s, f_number, focal_mm, captured_at, width, height` |
| measured (geometry) | `image_quad, px_per_image_px, measured_distance_m, measured_angle_deg, measured_azimuth_deg, measured_elevation_deg, reprojection_px, locate_ok` |

Side outputs: `<set>/check/<id>.jpg` (photo with screen and image quads drawn — **look at
these**) and `<set>/rectified/<id>.png` (the located image warped back to `image_px²`, the
input to channel measurement and decoding).

`locate_ok` requires inside/outside contrast on ≥ 3 sides *and* a PnP reprojection error
< 5 px — a located quad that is not a physically consistent square is a mis-location.
`measured_distance_m` scales with the configured pixel pitch; the angles do not.

## How the locator works (screen medium)

1. Panel = largest **uniform**, dark, low-saturation region on a 2k-wide downscale (local std
   < 8 over 15 px — the reflective wall and the desk fail this, the panel passes even with its
   oblique-angle glow). Holes (the image) are filled; convex hull → 4 corners.
2. The image's expected quad = the centred `image_px` square of the native display grid,
   projected through the panel homography.
3. Each side is snapped to the strongest colour edge within ±64 px (then ±16 px) in a canonical
   view, **admitting only positions whose outer band matches the surround colour** — this is
   what stops the snap landing on an interior text line or tree line.
4. If exactly one side still has no contrast (navy poster against oblique panel glow), it is
   placed from the other three using the fact that the image is a square in display space.
5. PnP on the four corners gives distance and viewing angle; reprojection error is the
   consistency check.

Print medium: not implemented until the print sheet with fiducials exists (Task 24).
