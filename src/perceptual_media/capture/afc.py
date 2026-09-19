"""Two-alternative forced choice (2AFC) visibility study (plan Task 26).

``pm-capture afc serve --sheet <name>`` starts a local page: each trial shows the original and
the marked version of one slide side by side at **1:1 device pixels** (the page divides by
``devicePixelRatio``), in random left/right order, for as long as the participant wants, and asks
"which one is the modified image?". Unlimited viewing time is the conservative choice: it is
the easiest possible condition for spotting the mark. Every slide is shown ``repeats`` times per
participant, in a random order seeded by the participant name and start time.

Responses append to ``<paths.captures>/2afc/<sheet>/responses.csv`` (irreplaceable data → the
external root). ``pm-capture afc report --sheet <name>`` prints accuracy per marker, per class
and per participant with exact (Clopper–Pearson) 95 % intervals and a two-sided binomial test
against chance (0.5). "At chance" — the invisibility claim — means the interval contains 0.5 and
its upper bound is small; a 75 % point is the usual detection threshold.

The page is served from this process only (``localhost``); no framework, no dependencies.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from scipy import stats

RESPONSE_COLUMNS = [
    "participant",
    "sheet",
    "slide_id",
    "image_id",
    "image_class",
    "marker",
    "strength",
    "trial",
    "marked_side",
    "choice",
    "correct",
    "rt_ms",
    "device_pixel_ratio",
    "timestamp",
]


@dataclass
class Study:
    sheet_dir: Path
    corpus: Path
    out_dir: Path
    repeats: int = 2

    def __post_init__(self) -> None:
        self.sheet = pd.read_csv(self.sheet_dir / "sheet.csv", keep_default_na=False)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.responses = self.out_dir / "responses.csv"

    def trials(self, participant: str) -> list[dict[str, object]]:
        rng = random.Random(f"{participant}|{time.time_ns()}")
        rows = []
        for _, r in self.sheet.iterrows():
            for _k in range(self.repeats):
                rows.append(
                    {
                        "slide_id": r["slide_id"],
                        "image_id": r["image_id"],
                        "image_class": r["image_class"],
                        "marker": r["marker"],
                        "strength": float(r["strength"]),
                        "marked_side": rng.choice(["left", "right"]),
                    }
                )
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            row["trial"] = i
        return rows

    def record(self, resp: dict[str, object]) -> None:
        new = not self.responses.exists()
        with open(self.responses, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=RESPONSE_COLUMNS)
            if new:
                w.writeheader()
            w.writerow({k: resp.get(k, "") for k in RESPONSE_COLUMNS})

    def image_path(self, slide_id: str, which: str) -> Path | None:
        rows = self.sheet[self.sheet["slide_id"] == slide_id]
        if rows.empty or which not in {"orig", "marked"}:
            return None
        r = rows.iloc[0]
        return self.corpus / r["original"] if which == "orig" else self.sheet_dir / f"{slide_id}_marked.png"


# ruff: noqa: E501 - the HTML/JS template below is one string; wrapping it hurts more than it helps
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>2AFC</title>
<style>
 body{background:#808080;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:16px;text-align:center}
 .pair{display:flex;justify-content:center;gap:48px;margin:24px 0}
 .pair img{display:block;image-rendering:pixelated;cursor:pointer;border:2px solid transparent}
 .pair img:hover{border-color:#fff}
 button{font-size:16px;padding:8px 16px} input{font-size:16px;padding:6px}
 #status{margin-top:12px;color:#ddd} .small{font-size:13px;color:#ccc}
</style></head><body>
<h2>Which image is the modified one?</h2>
<div id="start"><p>Sit about 60 cm from the screen. Take as long as you like on each pair.<br>
Click the image you think was modified. If you cannot tell, guess.</p>
<input id="name" placeholder="your name" onkeydown="if(event.key==='Enter')begin()"> <button onclick="begin()">Start</button> <span id="hint"></span></div>
<div id="trial" style="display:none">
 <div class="pair"><img id="left" onclick="answer('left')"><img id="right" onclick="answer('right')"></div>
 <div id="status"></div>
</div>
<div id="done" style="display:none"><h3>Done — thank you.</h3><p id="summary"></p></div>
<p class="small">Images are shown at 1:1 device pixels (devicePixelRatio = <span id="dpr"></span>).</p>
<script>
let trials=[], i=0, t0=0, name='';
const dpr=window.devicePixelRatio||1; document.getElementById('dpr').textContent=dpr.toFixed(2);
const px = %(image_px)d/dpr+'px';
for (const id of ['left','right']) { const el=document.getElementById(id); el.style.width=px; el.style.height=px; }
async function begin(){
  name=document.getElementById('name').value.trim();
  if(!name){document.getElementById('hint').textContent='enter a name first';return;}
  trials=await (await fetch('/trials?participant='+encodeURIComponent(name))).json();
  document.getElementById('start').style.display='none'; document.getElementById('trial').style.display='block';
  show();
}
function show(){
  if(i>=trials.length){finish();return;}
  const t=trials[i]; const l=t.marked_side==='left';
  document.getElementById('left').src='/img/'+t.slide_id+'/'+(l?'marked':'orig')+'.png?'+i;
  document.getElementById('right').src='/img/'+t.slide_id+'/'+(l?'orig':'marked')+'.png?'+i;
  document.getElementById('status').textContent='pair '+(i+1)+' of '+trials.length;
  t0=performance.now();
}
let correct=0;
async function answer(side){
  const t=trials[i]; const ok=(side===t.marked_side); correct+=ok?1:0;
  await fetch('/response',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(
    {...t, participant:name, choice:side, correct:ok, rt_ms:Math.round(performance.now()-t0), device_pixel_ratio:dpr})});
  i++; show();
}
function finish(){
  document.getElementById('trial').style.display='none'; document.getElementById('done').style.display='block';
  document.getElementById('summary').textContent=correct+' / '+trials.length+' correct (chance = '+Math.round(trials.length/2)+')';
}
</script></body></html>"""


def make_handler(study: Study, image_px: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            if url.path == "/":
                self._send((PAGE % {"image_px": image_px}).encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/trials":
                participant = parse_qs(url.query).get("participant", ["anon"])[0]
                self._send(json.dumps(study.trials(participant)).encode("utf-8"), "application/json")
            elif url.path.startswith("/img/"):
                parts = url.path.split("/")
                p = study.image_path(parts[2], parts[3].removesuffix(".png")) if len(parts) == 4 else None
                if p is None or not p.exists():
                    self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                else:
                    self._send(p.read_bytes(), "image/png")
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/response":
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            n = int(self.headers.get("Content-Length", "0"))
            resp = json.loads(self.rfile.read(n).decode("utf-8"))
            resp["sheet"] = study.sheet_dir.name
            resp["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            study.record(resp)
            self._send(b"ok", "text/plain")

    return Handler


def serve(study: Study, image_px: int, port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(study, image_px))
    print(f"2AFC study on http://127.0.0.1:{port}/  (responses -> {study.responses}); Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def binomial_summary(correct: int, n: int) -> dict[str, float]:
    """Accuracy with the exact 95 % Clopper–Pearson interval and a two-sided test against 0.5."""
    lo = stats.beta.ppf(0.025, correct, n - correct + 1) if correct > 0 else 0.0
    hi = stats.beta.ppf(0.975, correct + 1, n - correct) if correct < n else 1.0
    p = stats.binomtest(correct, n, 0.5).pvalue if n > 0 else float("nan")
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_vs_chance": p,
    }


def report(responses_csv: Path) -> pd.DataFrame:
    """Accuracy tables (overall, per marker, per marker × class, per participant); returns the per-marker × class table."""
    df = pd.read_csv(responses_csv, keep_default_na=False)
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1"])

    def table(by: list[str]) -> pd.DataFrame:
        rows = []
        for key, g in df.groupby(by):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(by, key, strict=True)), **binomial_summary(int(g["correct"].sum()), len(g))})
        return pd.DataFrame(rows)

    overall = binomial_summary(int(df["correct"].sum()), len(df))
    print(
        f"overall: {overall['correct']}/{overall['n']} = {overall['accuracy']:.3f} "
        f"[{overall['ci_lo']:.3f}, {overall['ci_hi']:.3f}]  p vs chance = {overall['p_vs_chance']:.3g}  "
        f"participants = {df['participant'].nunique()}"
    )
    fmt = {
        "accuracy": "{:.3f}".format,
        "ci_lo": "{:.3f}".format,
        "ci_hi": "{:.3f}".format,
        "p_vs_chance": "{:.3g}".format,
    }
    with pd.option_context("display.width", 200):
        for by in (["marker"], ["marker", "image_class"], ["participant"]):
            print(f"\nby {' x '.join(by)}:")
            print(table(by).to_string(index=False, formatters=fmt))
    return table(["marker", "image_class"])
