"""
Document numbers: many registers number their entries (N° 1, 2, 3, ...) and
the same number is written on the last page of the document ('No: 1.', often
a large calligraphic N° whose 'N°' the HTR turns into noise; sometimes after a
few blank pages; on a spread the number can stand on the left page while the
next document starts on the right page).

The marks are read from the PageXML, including regions without a type (the
plain-text export leaves those out):
  - explicit: 'No: 1.', 'N:o 7', 'N° 3' as a line of its own;
  - bare: a line with only a number (1-3 digits), outside page-number regions;
    its height relative to the page's text lines tells the large ones apart.
Per register, the monotone sequence of marks that matches its item numbers
(in scan order) is chosen by dynamic programming; the sequence removes most
noise (amounts, folio numbers). A matched mark 'N° k' ends entry k on its
scan; entry k+1 starts on the same scan when the mark is on the left page of a
spread with text on the right, else on the first scan with text after it.
"""

import os
import re
import zipfile

import numpy as np
import pandas as pd

from .registers import PAGEXML_DIR, page_lines

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MARK_DIR = os.path.join(DATA_DIR, "registers", "marks")
EXPLICIT = re.compile(r"^\W{0,3}N\W{0,3}[o°0]\W{0,3}\s*(\d{1,3})\W{0,3}$", re.I)
BARE = re.compile(r"^\W{0,2}(\d{1,3})\W{0,2}$")
WEIGHT_EXPLICIT, WEIGHT_LARGE, WEIGHT_BARE = 3.0, 1.5, 0.4
LARGE = 1.3  # bare numbers at least this many times the height of the page's text lines


def volume_marks(inventory: str, refresh: bool = False) -> pd.DataFrame:
    """Number marks of every scan of a volume (cached): filename, n, kind, weight, left (on the left page of a spread)."""
    path = os.path.join(MARK_DIR, f"{inventory}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    rows = []
    zpath = os.path.join(PAGEXML_DIR, f"{inventory}.zip")
    if os.path.exists(zpath):
        z = zipfile.ZipFile(zpath)
        for name in sorted(n for n in z.namelist() if n.endswith(".xml")):
            fn = os.path.basename(name)[:-4]
            try:
                lines, W, H = page_lines(z.read(name))
            except Exception:
                continue
            body = [l for l in lines if l["region"] in ("paragraph", "header")]
            med_h = float(np.median([l["y1"] - l["y0"] for l in body])) if body else 0.01
            spread = W > 1.2 * H
            side_lines = {True: sum(1 for l in body if l["x1"] < 0.5), False: sum(1 for l in body if l["x1"] >= 0.5)} if spread else {True: len(body), False: len(body)}
            for l in lines:
                t = l["text"].strip()
                m = EXPLICIT.match(t)
                kind = "explicit" if m else None
                if not m and l["region"] != "page-number":
                    m = BARE.match(t)
                    kind = "bare" if m else None
                if not m:
                    continue
                rel = (l["y1"] - l["y0"]) / max(med_h, 1e-3)
                left = bool(spread and l["x1"] < 0.5)
                text_lines = side_lines[left]
                if kind == "bare" and rel < LARGE and text_lines > 3 and l["region"] in ("paragraph", "marginalia", "header"):
                    continue  # a small number in running text or a list: not a document number
                weight = WEIGHT_EXPLICIT if kind == "explicit" else (WEIGHT_LARGE if rel >= LARGE else WEIGHT_BARE)
                rows.append({"filename": fn, "n": int(m.group(1)), "kind": kind, "weight": weight, "rel_height": round(rel, 2),
                             "left": left, "region": l["region"], "text_lines": text_lines})
    df = pd.DataFrame(rows, columns=["filename", "n", "kind", "weight", "rel_height", "left", "region", "text_lines"])
    os.makedirs(MARK_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def align(items: list[int], marks: pd.DataFrame) -> dict[int, int]:
    """
    {index into items: index into marks}: the best monotone matching of the
    register's item numbers (in register order) to marks with the same number,
    in scan order. At most one matched mark per page side (a document takes at
    least one), each item at most once.
    """
    if not len(items) or not len(marks):
        return {}
    marks = marks.reset_index(drop=True)
    units = list(marks.groupby(["pos", "left"], sort=False).indices.items())
    units.sort(key=lambda u: (u[0][0], not u[0][1]))  # scan order, left page before right
    n_i, n_u = len(items), len(units)
    best = np.zeros((n_i + 1, n_u + 1))
    pick = {}
    for i in range(1, n_i + 1):
        for u in range(1, n_u + 1):
            idx = units[u - 1][1]
            cand = [k for k in idx if marks.at[k, "n"] == items[i - 1]]
            v = max(best[i - 1, u], best[i, u - 1])
            if cand:
                k = max(cand, key=lambda k: marks.at[k, "weight"])
                t = best[i - 1, u - 1] + marks.at[k, "weight"]
                if t > v:
                    best[i, u], pick[(i, u)] = t, k
                    continue
            best[i, u] = v
    out, i, u = {}, n_i, n_u
    while i > 0 and u > 0:
        if (i, u) in pick and best[i, u] == best[i - 1, u - 1] + marks.at[pick[(i, u)], "weight"]:
            out[i - 1] = pick[(i, u)]
            i, u = i - 1, u - 1
        elif best[i - 1, u] >= best[i, u - 1]:
            i -= 1
        else:
            u -= 1
    return out
