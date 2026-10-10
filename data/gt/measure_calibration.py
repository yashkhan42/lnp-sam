"""Ground-truth calibration for the golden panels (spec 12.2): writes calibration.csv.

Independent of ``lnpsam.cleanup.calibrate``: each golden panel is an exact,
unresized crop of a source figure (checked below), so its scale bar is measured
in the figure, where the panel crop may have cut it off. The bar length is the
distance between the half-maximum edges of a line profile along the middle of
the bar (sub-pixel), as in an ImageJ line-profile measurement; the half-max
reference at each end is the local level just past that end. Scale values were
read by eye from the figures.

Usage (from the repo root):
    python data/gt/measure_calibration.py [--figures DIR] [--check-dir DIR]
``--check-dir`` writes zoomed crops of each bar with the measured ends in red.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SEM_WIN = (0, -45, 110, 25)

# panel_id -> (scale value in nm read by eye, bar polarity, search window).
# Window: (dx0, dy0, dx1, dy1) offsets from (panel left, panel bottom, panel right,
# panel bottom) in figure pixels, or ("abs", X0, Y0, X1, Y1) in figure pixels.
PANELS = {
    "pylypchuk2021_greenchem_04058d__p05_x46__panel02": (300, "bright", SEM_WIN),
    "pylypchuk2021_greenchem_04058d__p05_x46__panel17": (300, "bright", SEM_WIN),
    "pylypchuk2021_greenchem_04058d__p05_x46__panel01": (300, "bright", SEM_WIN),
    "pylypchuk2021_greenchem_04058d__p05_x46__panel04": (300, "bright", SEM_WIN),
    "pylypchuk2021_greenchem_04058d__p05_x46__panel19": (300, "bright", SEM_WIN),
    "pylypchuk2021_greenchem_04058d__p05_x46__panel11": (300, "bright", SEM_WIN),
    "pylypchuk2021_greenchem_04058d__p06_x50__panel04": (300, "bright", SEM_WIN),
    "girard2024_acssuschemeng_4c00711__p08_x658__panel06": (100, "bright", (0, -60, -150, 0)),
    # white bar inside the black "500nm" box, cut off by the panel crop
    "xiong2020_indcrop_112063__p02_x12__panel00": (500, "bright", ("abs", 280, 312, 372, 368)),
    # black bar at the top-left, a dark particle touches its left end
    "lievonen2016_greenchem_01436k__p04_x48__panel03": (100, "dark", ("abs_top", 0, 0, -300, 60)),
    # bracketed bar: measured over its full drawn extent, serif to serif
    "wang2019_holzforschung_0062__p08_x77__panel01": (500, "dark", (0, -110, -500, 0)),
    "pylypchuk2020_acssuschemeng_0c05119__p05_x218__panel02": (200, "dark", (300, -120, 0, 0)),
    "pylypchuk2021_greenchem_04058d__p05_x46__panel00": (300, "bright", SEM_WIN),
}


def locate(fig, panel):
    """Top-left of ``panel`` in ``fig``; asserts it is an exact, unresized crop."""
    res = cv2.matchTemplate(fig, panel, cv2.TM_SQDIFF)
    _, _, (x, y), _ = cv2.minMaxLoc(res)
    h, w = panel.shape
    if np.abs(fig[y:y + h, x:x + w] - panel).max() != 0:
        raise ValueError("panel is not an exact crop of its source figure")
    return x, y


def window(spec, x, y, w, h, shape):
    if spec[0] == "abs":
        X0, Y0, X1, Y1 = spec[1:]
    elif spec[0] == "abs_top":  # offsets from the panel's top edge
        _, a, b, c, d = spec
        X0, Y0, X1, Y1 = x + a, y + b, x + w + c, y + d
    else:
        a, b, c, d = spec
        X0, Y0, X1, Y1 = x + a, y + h + b, x + w + c, y + h + d
    return max(0, X0), max(0, Y0), min(shape[1], X1), min(shape[0], Y1)


def find_bar(win, polarity, px0, px1):
    """Longest bar-like component starting inside the panel's columns.

    Returns its slices and the 3 rows along its middle used for the profile."""
    mask = win >= 0.90 if polarity == "bright" else win <= 0.25
    white = win >= 0.80
    mask[white.mean(axis=1) > 0.9] = False  # white gutters between figure cells
    mask[:, white.mean(axis=0) > 0.9] = False
    lab, _ = ndimage.label(mask)
    best = None
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        inside = px0 <= sl[1].start < px1 and sl[1].start > 0 and sl[1].stop < win.shape[1]
        fill = (lab[sl] == i).mean()  # >= 0.3 keeps bracketed bars (end serifs)
        if inside and w >= 25 and w >= 4 * h and fill >= 0.3 and (best is None or w > best[1]):
            best = (sl, w, i)
    if best is None:
        raise RuntimeError("no scale bar found in the search window")
    sl, _, i = best
    runs = (lab[sl] == i).sum(axis=1)
    rows = np.nonzero(runs >= 0.8 * runs.max())[0] + sl[0].start
    mid = len(rows) // 2
    return sl, rows[max(0, mid - 1): mid + 2]


def half_max_length(win, sl, rows, polarity, dep=0.15):
    """Distance between the half-max edges of the profile along ``rows``.

    Each end: walk out from the centre until the profile leaves the bar level by
    more than ``dep``; the reference there is the most background-like value
    within 8 px past the end (clears a blurred edge, and stays local so an
    adjacent particle or box border is the reference, not the far background)."""
    x0, x1 = sl[1].start, sl[1].stop
    lo, hi = max(0, x0 - 12), min(win.shape[1], x1 + 12)
    prof = win[rows][:, lo:hi].mean(axis=0)
    xs = np.arange(lo, hi)
    sign = 1 if polarity == "bright" else -1
    bar = float(np.median(prof[(xs >= x0 + 0.2 * (x1 - x0)) & (xs < x1 - 0.2 * (x1 - x0))]))
    c = int(np.argmin(np.abs(xs - (x0 + x1) / 2)))
    edges = {}
    for step in (-1, 1):
        k = c
        while 0 <= k + step < len(prof) and sign * (bar - prof[k + step]) < dep:
            k += step
        idx = [i for i in range(k + step, k + 10 * step, step) if 0 <= i < len(prof)]
        half = 0.5 * (bar + float(prof[idx].min() if sign > 0 else prof[idx].max()))
        m = k
        while sign * (prof[m] - half) > 0:
            m += step
        a = m - step  # crossing lies between the inside sample a and the outside sample m
        sa, sm = sign * (prof[a] - half), sign * (prof[m] - half)
        edges[step] = xs[a] + step * sa / (sa - sm)  # in pixel-centre coordinates
    return float(edges[1] - edges[-1]), float(edges[-1]), float(edges[1])


def check_crop(win, sl, left, right, label, z=6, m=12):
    cy0, cy1 = max(0, sl[0].start - m), min(win.shape[0], sl[0].stop + m)
    cx0, cx1 = max(0, sl[1].start - m), min(win.shape[1], sl[1].stop + m)
    crop = cv2.cvtColor((win[cy0:cy1, cx0:cx1] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    for xe in (left, right):
        X = int(round((xe + 0.5 - cx0) * z))
        cv2.line(crop, (X, 0), (X, crop.shape[0] - 1), (0, 0, 255), 2)
    head = np.full((22, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(head, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([head, crop])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--figures", default="../lignin_segmentation/01_figures")
    ap.add_argument("--out", default=os.path.join(HERE, "calibration.csv"))
    ap.add_argument("--check-dir", default=None)
    args = ap.parse_args()

    rows = []
    for r in pd.read_csv(args.manifest).itertuples():
        scale_nm, polarity, spec = PANELS[r.panel_id]
        paper, fig_id = r.panel_id.split("__")[0], "__".join(r.panel_id.split("__")[:2])
        fig = cv2.imread(os.path.join(args.figures, paper, f"{fig_id}.png"), cv2.IMREAD_GRAYSCALE)
        fig = fig.astype(np.float32) / 255.0
        panel = cv2.imread(r.image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        x, y = locate(fig, panel)
        h, w = panel.shape
        X0, Y0, X1, Y1 = window(spec, x, y, w, h, fig.shape)
        win = fig[Y0:Y1, X0:X1]
        sl, prof_rows = find_bar(win, polarity, x - X0, x + w - X0)
        length, left, right = half_max_length(win, sl, prof_rows, polarity)
        bx0, bx1 = X0 + sl[1].start, X0 + sl[1].stop
        by0, by1 = Y0 + sl[0].start, Y0 + sl[0].stop
        rows.append({
            "panel_id": r.panel_id,
            "nm_per_px": round(scale_nm / length, 5),
            "scale_nm": scale_nm,
            "bar_px": round(length, 2),
            "bar_in_panel": bool(x <= bx0 and bx1 <= x + w and y <= by0 and by1 <= y + h),
            "source_figure": f"{fig_id}.png",
            "panel_xy_in_figure": f"{x},{y}",
        })
        if args.check_dir:
            os.makedirs(args.check_dir, exist_ok=True)
            label = f"{r.panel_id[-44:]}  {scale_nm} nm / {length:.2f} px"
            cv2.imwrite(os.path.join(args.check_dir, f"{r.panel_id}__bar.png"),
                        check_crop(win, sl, left, right, label))
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
