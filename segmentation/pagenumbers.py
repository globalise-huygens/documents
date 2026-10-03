"""
Page (folio) numbers that the layout analysis missed: a detector for the
number on the page, and TrOCR to read it.

The PageXML marks page numbers as regions of type 'page-number' (the source
of page.page_or_folio_number), but on many scans of the volumes without an
index the number is written yet not detected. Training data: scans where the
region was detected and holds digits, box = the region; validation on other
inventories. Images come from the IIIF service at TRAIN_WIDTH pixels.

  build_dataset()  → data/pagenumbers/{images,labels}/{train,val} (YOLO format) + dataset.yaml
  (training: ultralytics YOLO, see README; kept outside the project dependencies)
  read_numbers()   → detector + TrOCR on scans, number per scan
"""

import io
import logging
import os
import random
import re
import sqlite3
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from lxml import etree

logger = logging.getLogger("segmentation")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR = os.path.join(DATA_DIR, "pagenumbers")
PAGEXML_DIR = os.environ.get("SEGMENTATION_PAGEXML_DIR", "/Volumes/HDE0090")
NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
TRAIN_WIDTH = 1280


def number_boxes(xml: bytes) -> tuple[list[tuple[float, float, float, float, str]], float, float]:
    """Page-number regions with digits: (x0, y0, x1, y1) as shares of the image, and their text."""
    root = etree.fromstring(xml)
    page = root.find("p:Page", NS)
    W, H = float(page.get("imageWidth") or 1), float(page.get("imageHeight") or 1)
    out = []
    for region in root.iterfind(".//p:TextRegion", NS):
        if "page-number" not in (region.get("custom") or ""):
            continue
        text = " ".join((l.find("p:TextEquiv/p:Unicode", NS).text or "") for l in region.iterfind("p:TextLine", NS) if l.find("p:TextEquiv/p:Unicode", NS) is not None)
        if not re.search(r"\d", text):
            continue
        xy = np.array([[float(v) for v in p.split(",")] for p in region.find("p:Coords", NS).get("points").split()])
        out.append((xy[:, 0].min() / W, xy[:, 1].min() / H, xy[:, 0].max() / W, xy[:, 1].max() / H, text.strip()))
    return out, W, H


def _iiif(info: str, width: int) -> str:
    return f"{info.rsplit('/info.json', 1)[0]}/full/{width},/0/default.jpg"


def _fetch(url: str, tries: int = 3) -> bytes | None:
    import requests

    for k in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.ok:
                return r.content
        except Exception:
            pass
        time.sleep(1 + k)
    return None


def build_dataset(conn: sqlite3.Connection, target_inventories: list[str], n_train: int = 3500, n_val: int = 700, seed: int = 0, workers: int = 8):
    """Scans with a detected page number, half from the target inventories; validation inventories are disjoint."""
    rng = random.Random(seed)
    rows = conn.execute(
        "SELECT i.inventory_number, s.filename, s.iiif_image_info FROM page p JOIN scan s ON s.id = p.scan_id "
        "JOIN inventory i ON i.id = s.inventory_id WHERE p.page_or_folio_number IS NOT NULL AND p.page_or_folio_number <> '' "
        "AND s.iiif_image_info IS NOT NULL GROUP BY s.id"
    ).fetchall()
    by_inv: dict[str, list] = {}
    for inv, fn, info in rows:
        by_inv.setdefault(inv, []).append((fn, info))
    targets = set(target_inventories)
    invs = sorted(by_inv)
    rng.shuffle(invs)
    val_invs = set(invs[: max(1, len(invs) // 6)])

    def draw(pool_invs, n):
        t = [i for i in pool_invs if i in targets]
        o = [i for i in pool_invs if i not in targets]
        picks = []
        for group, k in ((t, n // 2), (o, n - n // 2)):
            for _ in range(k):  # a few scans from many inventories
                inv = rng.choice(group)
                picks.append((inv,) + rng.choice(by_inv[inv]))
        return picks

    jobs = [("val", *x) for x in draw([i for i in invs if i in val_invs], n_val)] + [("train", *x) for x in draw([i for i in invs if i not in val_invs], n_train)]
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)
    zips: dict[str, zipfile.ZipFile] = {}

    def one(job):
        split, inv, fn, info = job
        img_path = os.path.join(OUT_DIR, "images", split, fn + ".jpg")
        if os.path.exists(img_path):
            return True
        try:
            z = zips.get(inv) or zipfile.ZipFile(os.path.join(PAGEXML_DIR, f"{inv}.zip"))
            zips[inv] = z
            name = next(n for n in z.namelist() if n.endswith(fn + ".xml"))
            boxes, _, _ = number_boxes(z.read(name))
        except Exception:
            return False
        if not boxes:
            return False
        data = _fetch(_iiif(info, TRAIN_WIDTH))
        if data is None:
            return False
        with open(img_path, "wb") as f:
            f.write(data)
        with open(os.path.join(OUT_DIR, "labels", split, fn + ".txt"), "w") as f:
            for x0, y0, x1, y1, _ in boxes:
                f.write(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n")
        return True

    t0, ok = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, jobs), 1):
            ok += bool(r)
            if k % 250 == 0 or k == len(jobs):
                logger.info("  page-number dataset: %d/%d scans (%d usable, %.0fs; about %.0f min to go)", k, len(jobs), ok, time.time() - t0,
                            (time.time() - t0) / k * (len(jobs) - k) / 60)
    with open(os.path.join(OUT_DIR, "dataset.yaml"), "w") as f:
        f.write(f"path: {OUT_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: page-number\n")
    logger.info("Page-number dataset in %s (%d scans)", OUT_DIR, ok)
