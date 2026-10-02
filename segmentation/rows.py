"""
Entries of handwritten registers, rebuilt from the PageXML line geometry.

The HTR splits a register row into several line fragments wherever the
writing leaves wide gaps (dittos, dotted leaders), and the reading order of
those fragments mixes rows. Here:

1. Rows: fragments whose baselines are at about the same height (within half
   a typical line height) and that do not overlap horizontally form one row,
   ordered left to right.
2. Entries: a row that starts in the key column (a folio range, item number
   or packet mark at the left) starts an entry; an indented row without one
   continues the entry above it. Rows above the first entry are the
   register's heading.
3. Fields: folio range ('fo. 69 a 112', '115 „ 122'), item number ('N:o 3'),
   packet marks and 'ingenaaijt' / 'niet ontfangen', date ('in dato 10 Jan
   1775'), and the description.
4. Dittos: an entry whose description is mostly 'dito' / 'd:o' / '„' takes
   the description of the entry above it, with its own date and other words.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from .registers import page_lines, pagexml_names

DITTO_TOKEN = re.compile(r"^(dito|d[:\.]?o|„+|_:?o|idem|\.|-|—|–|,)+$", re.I)
FOLIO = re.compile(r"^\W{0,3}(?:f\W{0,3}o?\W{0,3})?(\d{1,4})\b[\s.:]*(?:[„\"'aà,\-–]|tot)?[\s.:]*(\d{1,4})\b|^\W{0,3}(?:f\W{0,3}o?\W{0,3})?(\d{1,4})\b", re.I)
ITEM = re.compile(r"\bN\W{0,2}[o°]\W{0,3}(\d{1,3})\b|^\W{0,2}„\s*(\d{1,3})\b")
PACKET = re.compile(r"^((?:[A-Z]\W{0,2}){1,8})(?=\s|$)")
SEWN = re.compile(r"inge?na[aeoi]{0,2}[iy]?[jy]?[dt]|niet\s+ontfangen|in\s+de\s+kas", re.I)
MONTH = r"(?:jan\w*|feb\w*|f[ae]b\w*|ma[ae]?r\w*|apr\w*|m[ae][iy]j?\w*|maij|jun\w*|jul\w*|aug\w*|sep\w*|7ber|oct\w*|okt\w*|8ber|nov\w*|9ber|dec\w*|xber|10ber|ult\W{0,2}o?)"
DATE = re.compile(r"\b(?:in|de|sub)\s*dato\s+(.{3,40}?\b1[67]\d\d?\b|.{3,30}$)|\b(\d{1,2}\W{0,3}\s*" + MONTH + r"\W{0,3}\s*1[67][\dor]{1,2})\b", re.I)
ANY_DATE = re.compile(r"\b(?:(?:in|de|sub)\s*dato\s+)?\d{1,2}\W{0,3}\s*" + MONTH + r"\W{0,3}\s*1[67][\dor]{1,2}\b\W?|\b(?:in|de|sub)\s*dato\b", re.I)
KEY = re.compile(r"^\W{0,3}(f\W{0,3}o?\W{0,3}\d|\d{1,4}\b|N\W{0,2}[o°]|(?:[A-Z]\W{1,2}){1,6}(\s|$))")


@dataclass
class Row:
    cells: list[dict]  # line fragments, left to right

    @property
    def x0(self) -> float:
        return self.cells[0]["x0"]

    @property
    def y(self) -> float:
        return float(np.median([c["y"] for c in self.cells]))

    @property
    def text(self) -> str:
        return " ".join(c["text"] for c in self.cells)


@dataclass
class Entry:
    rows: list[Row] = field(default_factory=list)
    folio_start: int | None = None
    folio_end: int | None = None
    item: str | None = None
    packet: str | None = None
    sewn: str | None = None
    date: str | None = None
    description: str = ""
    resolved: str = ""
    dittos: int = 0
    leading_ditto: bool = False  # starts with 'dito': same kind of document as the entry above

    @property
    def raw(self) -> str:
        return " / ".join(r.text for r in self.rows)


def build_rows(lines: list[dict]) -> list[Row]:
    """Group line fragments into rows by baseline height."""
    if not lines:
        return []
    heights = [l["y1"] - l["y0"] for l in lines if l["y1"] > l["y0"]]
    tol = 0.5 * float(np.median(heights)) if heights else 0.008
    tol = min(max(tol, 0.004), 0.015)
    rows: list[list[dict]] = []
    for l in sorted(lines, key=lambda l: l["y"]):
        placed = False
        for r in reversed(rows[-3:]):  # only the most recent rows can share the height
            ry = float(np.median([c["y"] for c in r]))
            overlaps = any(min(l["x1"], c["x1"]) - max(l["x0"], c["x0"]) > 0.3 * min(l["x1"] - l["x0"], c["x1"] - c["x0"]) for c in r)
            if abs(l["y"] - ry) <= tol and not overlaps:
                r.append(l)
                placed = True
                break
        if not placed:
            rows.append([l])
    return [Row(sorted(r, key=lambda c: c["x0"])) for r in rows]


def _key_column(rows: list[Row]) -> float | None:
    xs = [r.x0 for r in rows if KEY.match(r.cells[0]["text"])]
    return float(np.median(xs)) if len(xs) >= 2 else None


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text) if w]


def parse_entry(e: Entry):
    text = e.raw.replace(" / ", " ")
    m = SEWN.search(text)
    if m:
        e.sewn = m.group(0)
        text = text[: m.start()] + " " + text[m.end():]
    m = FOLIO.match(text)
    if m:
        if m.group(1):
            e.folio_start, e.folio_end = int(m.group(1)), int(m.group(2))
        else:
            e.folio_start = e.folio_end = int(m.group(3))
        text = text[m.end():]
    m = ITEM.search(text)
    if m:
        e.item = m.group(1) or m.group(2)
        text = text[: m.start()] + " " + text[m.end():]
    m = PACKET.match(text.strip())
    if m and len(m.group(1).replace(" ", "")) >= 2:
        e.packet = m.group(1).strip()
        text = text.strip()[m.end():]
        m2 = re.match(r"^\W{0,12}(\d{1,3})\b", text)  # 'Z. A. D. R. H. E. „ 2.': item number after the packet mark
        if m2 and e.item is None:
            e.item = m2.group(1)
            text = text[m2.end():]
    e.leading_ditto = bool(re.match(r"^\W*(dito|d[:\.]?o|_:?o|idem)\b", text.strip(), re.I))
    m = DATE.search(text)
    if m:
        e.date = (m.group(1) or m.group(2)).strip(" .,")
    words = _words(text)
    e.dittos = sum(bool(DITTO_TOKEN.match(w)) for w in words)
    e.description = " ".join(w for w in words if not DITTO_TOKEN.match(w)).strip(" .,-")


def resolve_dittos(entries: list[Entry]):
    """An entry that is mostly dittos takes the description of the entry above (its own words added)."""
    prev = ""  # description of the entry above, without its date
    for e in entries:
        own = re.sub(r"\s+", " ", ANY_DATE.sub(" ", e.description)).strip(" .,-")
        if e.dittos and prev and (e.dittos >= len(_words(own)) or len(_words(own)) <= 4):
            e.resolved = " ".join(x for x in (prev, own, f"in dato {e.date}" if e.date else "") if x)
        elif e.leading_ditto and prev:  # 'dito door als eevengemeld, gerigt aan ...'
            head = " ".join(_words(prev)[:8])
            e.resolved = " ".join(x for x in (f"Idem ({head} …)", own, f"in dato {e.date}" if e.date else "") if x)
        else:
            e.resolved = " ".join(x for x in (own, f"in dato {e.date}" if e.date else "") if x)
            prev = own or prev


def split_columns(lines: list[dict]) -> list[list[dict]]:
    """Two columns of entries side by side: lines starting left and right of a clear
    vertical gap, the left ones ending before the gap. Otherwise one column."""
    long = [l for l in lines if len(l["text"]) >= 8]
    if len(long) < 8:
        return [lines]
    xs = np.sort([l["x0"] for l in long])
    best = None
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2
        if b - a > 0.12 and 0.3 < mid < 0.7 and (not best or b - a > best[1]):
            best = (mid, b - a)
    if not best:
        return [lines]
    thr = best[0]
    left = [l for l in lines if l["x0"] < thr]
    right = [l for l in lines if l["x0"] >= thr]
    left_long = [l for l in left if len(l["text"]) >= 8]
    right_long = [l for l in right if len(l["text"]) >= 8]
    if right_long and np.mean([bool(ANY_DATE.search(l["text"])) for l in right_long]) >= 0.5:
        return [lines]  # a column of dates belongs to the entries on the left
    if min(len(left_long), len(right_long)) < 0.25 * len(long) or np.median([l["x1"] for l in left_long]) > np.median([l["x0"] for l in right_long]) + 0.02:
        return [lines]  # one column (a date column at the right is not a second column of entries)
    return [left, right]


def register_entries(lines: list[dict]) -> tuple[list[Row], list[Entry]]:
    """(heading rows, entries) of one register page; two columns are read left, then right."""
    heading, entries = [], []
    for k, column in enumerate(split_columns(lines)):
        h, e = _column_entries(column)
        if k == 0 or not entries:
            heading += h
        elif h and entries:  # rows at the top of the right column without a key continue the left column's last entry
            entries[-1].rows += h
            parse_entry(entries[-1])
        entries += e
    resolve_dittos(entries)
    return heading, entries


def _column_entries(lines: list[dict]) -> tuple[list[Row], list[Entry]]:
    rows = build_rows(lines)
    key_x = _key_column(rows)
    heading, entries = [], []
    for r in rows:
        first = r.cells[0]["text"]
        starts = bool(KEY.match(first)) and (key_x is None or r.x0 <= key_x + 0.05)
        if entries and len(re.findall(r"[^\W\d_]", r.text)) < 3 and not starts:
            entries[-1].rows.append(r)  # a stray fragment ('„', '&'): part of the entry above
        elif starts:
            entries.append(Entry([r]))
        elif entries and (key_x is None or r.x0 > key_x + 0.02):
            entries[-1].rows.append(r)  # an indented row continues the entry above
        elif entries:
            entries.append(Entry([r]))
        else:
            heading.append(r)
    for e in entries:
        parse_entry(e)
    return heading, entries


def scan_entries(filename: str) -> tuple[list[Row], list[Entry]]:
    inv = filename.split("_")[2]
    z, names = pagexml_names(inv)
    if z is None or filename not in names:
        return [], []
    return register_entries(page_lines(z.read(names[filename]))[0])


def _scan_range(first: str, last: str) -> list[str]:
    prefix = first.rsplit("_", 1)[0]
    a, b = int(first.rsplit("_", 1)[1]), int(last.rsplit("_", 1)[1])
    return [f"{prefix}_{k:04d}" for k in range(a, b + 1)]


def _range_entries(args) -> list[dict]:
    """Entries of every scan of one register range (one task per range)."""
    inventory, first, last, score = args
    z, names = pagexml_names(inventory)
    out = []
    if z is None:
        return out
    heading = []
    k = 0
    for fn in _scan_range(first, last):
        if fn not in names:
            continue
        try:
            head, entries = register_entries(page_lines(z.read(names[fn]))[0])
        except Exception:
            continue
        if not out:
            heading = [r.text for r in head]  # the heading of the register: rows above its first entry
        for e in entries:
            out.append({
                "inventory": inventory, "register_first_scan": first, "register_last_scan": last, "register_score": score,
                "register_heading": " / ".join(heading), "scan": fn, "entry": k, "folio_start": e.folio_start, "folio_end": e.folio_end,
                "item": e.item, "packet": e.packet, "sewn": e.sewn, "date": e.date, "dittos": e.dittos, "title": e.resolved, "raw": e.raw,
            })
            k += 1
    _items_or_folios(out)
    return out


def _items_or_folios(rows: list[dict]):
    """Single numbers that mostly run up by one are item numbers, not folios."""
    single = [r for r in rows if r["folio_start"] is not None and r["folio_start"] == r["folio_end"]]
    if len(single) < 3:
        return
    steps = np.diff([r["folio_start"] for r in single])
    if np.mean((steps >= 1) & (steps <= 2)) >= 0.6 and len(single) >= 0.5 * sum(r["folio_start"] is not None for r in rows):
        for r in single:
            if r["item"] is None:
                r["item"] = str(r["folio_start"])
            r["folio_start"] = r["folio_end"] = None


def entries_for_ranges(ranges, min_score: float = 0.9, workers: int = 6):
    """Register entries of all ranges scoring >= min_score, with progress."""
    import logging
    import time
    from multiprocessing import Pool

    import pandas as pd

    logger = logging.getLogger("segmentation")
    todo = ranges[ranges["max_score"] >= min_score]
    tasks = list(zip(todo["inventory"].astype(str), todo["first_scan"], todo["last_scan"], todo["max_score"]))
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_range_entries, tasks, chunksize=8), 1):
            rows += r
            if k % 500 == 0 or k == len(tasks):
                logger.info("  %d/%d register ranges, %d entries (%.0fs)", k, len(tasks), len(rows), time.time() - t0)
    return pd.DataFrame(rows)
