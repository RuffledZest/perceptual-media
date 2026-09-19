import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from perceptual_media.capture.afc import Study, binomial_summary, make_handler, report


@pytest.fixture
def study(tmp_path):
    sheet = tmp_path / "sheet"
    corpus = tmp_path / "corpus"
    (corpus / "x").mkdir(parents=True)
    sheet.mkdir()
    Image.fromarray(np.full((16, 16, 3), 100, np.uint8)).save(corpus / "x" / "a.png")
    Image.fromarray(np.full((16, 16, 3), 101, np.uint8)).save(sheet / "S00_marked.png")
    pd.DataFrame(
        [
            {
                "slide_id": "S00",
                "image_id": "a",
                "image_class": "t",
                "marker": "null",
                "strength": 1.0,
                "original": "x/a.png",
            }
        ]
    ).to_csv(sheet / "sheet.csv", index=False)
    return Study(sheet, corpus, tmp_path / "out", repeats=3)


def test_trials_are_balanced_and_shuffled(study):
    t = study.trials("p")
    assert len(t) == 3 and {r["marked_side"] for r in t} <= {"left", "right"}
    assert [r["trial"] for r in t] == [0, 1, 2]


def test_server_roundtrip_records_responses(study):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(study, 16))
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert b"Which image" in urllib.request.urlopen(base + "/").read()
        trials = json.loads(urllib.request.urlopen(base + "/trials?participant=zed").read())
        assert len(trials) == 3
        for which in ("orig", "marked"):
            png = urllib.request.urlopen(f"{base}/img/S00/{which}.png").read()
            assert png[:8] == b"\x89PNG\r\n\x1a\n"
        for t in trials:
            body = json.dumps(
                {
                    **t,
                    "participant": "zed",
                    "choice": t["marked_side"],
                    "correct": True,
                    "rt_ms": 5,
                    "device_pixel_ratio": 1,
                }
            ).encode()
            req = urllib.request.Request(
                base + "/response", data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            assert urllib.request.urlopen(req).read() == b"ok"
    finally:
        server.shutdown()
        server.server_close()
    df = pd.read_csv(study.responses)
    assert len(df) == 3 and df["correct"].all() and (df["sheet"] == "sheet").all()
    table = report(study.responses)
    assert table.loc[0, "correct"] == 3 and table.loc[0, "accuracy"] == 1.0


def test_binomial_summary_chance_and_ci():
    s = binomial_summary(50, 100)
    assert s["accuracy"] == 0.5 and s["ci_lo"] < 0.5 < s["ci_hi"] and s["p_vs_chance"] > 0.9
    s = binomial_summary(90, 100)
    assert s["ci_lo"] > 0.8 and s["p_vs_chance"] < 1e-10
    assert binomial_summary(0, 10)["ci_lo"] == 0.0 and binomial_summary(10, 10)["ci_hi"] == 1.0
