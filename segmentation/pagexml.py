"""
Page layout per scan from the PageXML files (one zip per inventory,
<SEGMENTATION_PAGEXML_DIR>/<inventory>.zip, e.g. /Volumes/HDE0090/1056.zip).

Laypa marks regions as paragraph, marginalia, header, catch-word, page-number,
signature-mark; every text line has a baseline. From the lines of the
paragraph regions, ordered top to bottom, we derive where on the page text
starts and stops, gaps between blocks of text, closing and opening formulas
halfway down the page (a document ending and the next starting on the same
page), centred headings, and catch-words (the text continues on the next page).

Each inventory is read once; the per-scan summary is cached as
data/layout/<inventory>.parquet. Without the zip (or the setting), the layout
features are missing (NaN) and count as neutral in the models.
"""

import os
import re
import zipfile

import numpy as np
import pandas as pd
from lxml import etree

PAGEXML_DIR = os.environ.get("SEGMENTATION_PAGEXML_DIR")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "layout")
NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
TYPE_RE = re.compile(r"type:([^;]+);")

LAYOUT_COLUMNS = [
    "catch_word",
    "text_top",
    "text_bottom",
    "max_line_gap",
    "closing_mid_page",
    "opening_below_top",
    "heading_below_top",
    "n_paragraph_lines",
]


def _lines(xml: bytes) -> tuple[list[tuple], float, float, set]:
    """(region type, y, x_min, x_max, text) per text line, page width, height, region types."""
    root = etree.fromstring(xml)
    page = root.find("p:Page", NS)
    width, height = float(page.get("imageWidth") or 1), float(page.get("imageHeight") or 1)
    out, types = [], set()
    for region in root.iterfind(".//p:TextRegion", NS):
        m = TYPE_RE.search(region.get("custom") or "")
        rtype = m.group(1) if m else ""
        types.add(rtype)
        for line in region.iterfind("p:TextLine", NS):
            base = line.find("p:Baseline", NS)
            pts = base.get("points") if base is not None else None
            if not pts:
                coords = line.find("p:Coords", NS)
                pts = coords.get("points") if coords is not None else None
            if not pts:
                continue
            xy = np.array([[float(v) for v in p.split(",")] for p in pts.split()])
            uni = line.find("p:TextEquiv/p:Unicode", NS)
            out.append((rtype, xy[:, 1].mean(), xy[:, 0].min(), xy[:, 0].max(), (uni.text or "") if uni is not None else ""))
    return out, width, height, types


def layout_features(xml: bytes) -> dict:
    from .predictors import CLOSING_RE, GENRE_RE, SALUTATION_RE  # avoid an import cycle

    lines, width, height, types = _lines(xml)
    feats = dict.fromkeys(LAYOUT_COLUMNS, 0.0)
    feats["catch_word"] = float("catch-word" in types)
    para = sorted((l for l in lines if l[0] in ("paragraph", "")), key=lambda l: l[1])
    feats["n_paragraph_lines"] = float(len(para))
    if not para:
        return feats
    ys = np.array([l[1] for l in para]) / height
    feats["text_top"] = float(ys[0])
    feats["text_bottom"] = float(ys[-1])
    if len(ys) >= 3:
        gaps = np.diff(ys)
        typical = np.median(gaps) or 1e-3
        feats["max_line_gap"] = float(min(gaps.max() / typical, 10.0))
    x0 = np.array([l[2] for l in para]) / width
    x1 = np.array([l[3] for l in para]) / width
    left, right = np.percentile(x0, 10), np.percentile(x1, 90)
    column = max(right - left, 1e-3)
    for k, (_, _, _, _, text) in enumerate(para):
        below = len(para) - 1 - k
        if CLOSING_RE.search(text) and ys[k] < 0.85 and below >= 3:
            feats["closing_mid_page"] = 1.0
        if ys[k] > 0.2 and (GENRE_RE.match(text.strip()) or SALUTATION_RE.search(text[:40])):
            feats["opening_below_top"] = 1.0
        w = (x1[k] - x0[k]) / column
        centre = ((x0[k] + x1[k]) / 2 - left) / column
        if ys[k] > 0.2 and w < 0.45 and 0.3 < centre < 0.7 and len(text.split()) >= 2:
            feats["heading_below_top"] = 1.0
    return feats


def load_layout(inventory_number: str, filenames: pd.Series) -> pd.DataFrame:
    """Layout features per scan (aligned with `filenames`); NaN where unavailable,
    which the models treat as neutral (the training average)."""
    empty = pd.DataFrame(np.nan, index=filenames.index, columns=LAYOUT_COLUMNS)
    empty["has_layout"] = 0.0
    cache = os.path.join(CACHE_DIR, f"{inventory_number}.parquet")
    if os.path.exists(cache):
        table = pd.read_parquet(cache)
    else:
        path = os.path.join(PAGEXML_DIR, f"{inventory_number}.zip") if PAGEXML_DIR else None
        if not path or not os.path.exists(path):
            return empty
        rows = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".xml"):
                    try:
                        rows.append({"filename": os.path.basename(name)[:-4], **layout_features(z.read(name))})
                    except (etree.XMLSyntaxError, ValueError):
                        continue
        table = pd.DataFrame(rows, columns=["filename"] + LAYOUT_COLUMNS)
        os.makedirs(CACHE_DIR, exist_ok=True)
        table.to_parquet(cache, index=False)
    table = table.set_index("filename")
    out = empty.copy()
    known = filenames.isin(table.index)
    out.loc[known, LAYOUT_COLUMNS] = table.loc[filenames[known], LAYOUT_COLUMNS].to_numpy()
    out.loc[known, "has_layout"] = 1.0
    return out
