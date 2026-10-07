"""
Hand review of the segmentation output (/review/segments in the app).

segments.csv (1 GB) is split once into data/segments/<inventory>.parquet, so
the app reads one inventory at a time; the split is redone when segments.csv
changes (split_segments, or `python -m segmentation review-split`).

The ToC shown for an inventory merges
  - the model's segments in scan order: ToC entries (one item per index entry,
    also when several entries share a span), derived entries, General
    Missives, court cases, unindexed documents and subdocuments; runs of
    non-document scans as gaps;
  - index entries the model did not place (kind 'unplaced'), after the
    placed entry that precedes them in the index;
  - documents added by reviewers (kind 'added') and reviewed items that are
    no longer in the model output (stale), at their reviewed start scan.

Reviews go to the segment_review table (models.SegmentReview), append-only.
"""

import json
import os
import re
import threading
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTS_CSV = os.environ.get("SEGMENTS_CSV", os.path.join(ROOT, "segments.csv"))
SEGMENT_DIR = os.path.join(ROOT, "data", "segments")
MARKER = "_source.json"

VERDICTS = {"not_document", "merge_previous", "not_in_inventory", "unsure"}
SIDES = {"left", "right"}


# ── split ─────────────────────────────────────────────────────────────────────


def _source_stamp(source: str) -> dict:
    st = os.stat(source)
    return {"path": os.path.abspath(source), "mtime": st.st_mtime, "size": st.st_size}


def split_is_current(source: str = SEGMENTS_CSV, target: str = SEGMENT_DIR) -> bool:
    try:
        with open(os.path.join(target, MARKER)) as f:
            return json.load(f) == _source_stamp(source)
    except (OSError, ValueError):
        return False


def split_segments(source: str = SEGMENTS_CSV, target: str = SEGMENT_DIR, chunksize: int = 50_000) -> int:
    """Write one parquet per inventory. The rows of an inventory are
    contiguous in segments.csv, so an inventory is written as soon as the
    next one starts. Returns the number of inventories."""
    os.makedirs(target, exist_ok=True)
    stamp = _source_stamp(source)
    marker = os.path.join(target, MARKER)
    if os.path.exists(marker):
        os.remove(marker)
    pending, n = None, 0

    def write(df):
        inv = df["inventory"].iat[0]
        tmp = os.path.join(target, f".{inv}.parquet")
        df.reset_index(drop=True).to_parquet(tmp, index=False)
        os.replace(tmp, os.path.join(target, f"{inv}.parquet"))

    for chunk in pd.read_csv(source, dtype=str, chunksize=chunksize, keep_default_na=False):
        if pending is not None:
            chunk = pd.concat([pending, chunk])
        breaks = chunk["inventory"].ne(chunk["inventory"].shift()).cumsum()
        groups = [g for _, g in chunk.groupby(breaks, sort=False)]
        for g in groups[:-1]:
            write(g)
            n += 1
        pending = groups[-1]
    if pending is not None and len(pending):
        write(pending)
        n += 1
    with open(marker, "w") as f:
        json.dump(stamp, f)
    return n


_split_lock = threading.Lock()
_split_error: list[str] = []


def ensure_split_async() -> str:
    """'ready', 'running' (a split started in the background) or 'error: ...'."""
    if split_is_current():
        return "ready"
    if not os.path.exists(SEGMENTS_CSV):
        return f"error: {SEGMENTS_CSV} not found"
    if _split_error:
        return "error: " + _split_error[-1]
    if _split_lock.locked():
        return "running"

    def run():
        with _split_lock:
            try:
                split_segments()
            except Exception as e:  # shown on the next request
                _split_error.append(str(e))

    threading.Thread(target=run, daemon=True).start()
    return "running"


def has_segments(inventory_number: str) -> bool:
    return os.path.exists(os.path.join(SEGMENT_DIR, f"{inventory_number}.parquet"))


# ── items ─────────────────────────────────────────────────────────────────────


def scan_number(filename: str | None) -> int | None:
    m = re.search(r"_(\d+)$", filename or "")
    return int(m.group(1)) if m else None


def _ids(raw: str) -> list[str]:
    return [x for x in (raw or "").split(";") if x]


def _evidence(raw: str) -> dict:
    try:
        ev = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    start, end = ev.get("start") or {}, ev.get("end") or {}
    return {
        "p_start": start.get("p_start"),
        "p_end": end.get("p_end"),
        "text_start": start.get("text_start") or "",
        "text_end": end.get("text_end") or "",
        "why_start": start.get("why_start") or [],
        "why_end": end.get("why_end") or [],
        "formulas": start.get("formulas") or "",
        "best_toc_match": start.get("best_toc_match") or "",
    }


def _folios(start, end) -> str:
    if pd.isna(start):
        return ""
    s, e = int(start), (int(end) if not pd.isna(end) else None)
    return f"fol. {s}" if e is None or e == s else f"fol. {s}–{e}"


def load_scans(conn, inventory_number: str) -> tuple[str, list[dict]]:
    """(inventory id, scans in physical order with IIIF base and folio numbers per side)."""
    row = conn.execute("SELECT id FROM inventory WHERE inventory_number = ?", (inventory_number,)).fetchone()
    if row is None:
        raise KeyError(inventory_number)
    inv_id = row[0]
    rows = conn.execute(
        "SELECT s.filename, s.scan_type, s.iiif_image_info, "
        "       group_concat(coalesce(p.recto_verso, '') || ':' || coalesce(p.page_or_folio_number, ''), '|') "
        "FROM scan s LEFT JOIN page p ON p.scan_id = s.id "
        "WHERE s.inventory_id = ? GROUP BY s.id ORDER BY s.scan_order, s.filename",
        (inv_id,),
    ).fetchall()
    scans = []
    for filename, scan_type, info, pages in rows:
        sides = {}
        for part in (pages or "").split("|"):
            side, _, folio = part.partition(":")
            if folio.strip():
                sides[side or "page"] = folio.strip()
        scans.append({
            "f": filename,
            "n": scan_number(filename),
            "double": scan_type == "Double",
            "iiif": info.rsplit("/info.json", 1)[0] if info else None,
            # verso before recto: the left and right page of a double scan
            "folio": " · ".join(f"{side[0].lower()} {v}" if side != "page" else v for side, v in sorted(sides.items(), reverse=True)),
        })
    return inv_id, scans


def model_items(seg: pd.DataFrame, toc: pd.DataFrame) -> list[dict]:
    """The model's segments as review items, in scan order."""
    by_id = {str(int(r.csv_id)): r for r in toc.itertuples() if not pd.isna(r.csv_id)}
    items = []
    for r in seg.itertuples():
        base = {"start": r.start_scan, "end": r.end_scan, "model_start": r.start_scan, "model_end": r.end_scan,
                "boundary": r.boundary, "segment": int(r.segment)}
        if r.kind == "non-document":
            items.append({**base, "key": f"gap:{r.start_scan}", "kind": "gap", "depth": 0, "title": ""})
            continue
        ev = _evidence(r.evidence)
        depth = 1 if any(getattr(r, c) for c in ("parent_csv_id", "parent_entry_id", "parent_court_case", "parent_gm_id")) else 0
        common = {**base, **ev, "depth": depth, "gm": bool(r.gm_ids), "date": r.date_begin or ""}
        csv_ids = _ids(r.toc_csv_ids)
        if r.kind == "toc" and csv_ids:
            placed = _ids(r.placed_by)
            for k, cid in enumerate(csv_ids):
                e = by_id.get(cid)
                items.append({
                    **common, "key": f"toc:{cid}", "kind": "toc", "csv_id": cid,
                    "title": (e.title if e is not None else None) or r.toc_title or r.title,
                    "index_folios": _folios(e.folio_start, e.folio_end) if e is not None else "",
                    "toc_order": float(e.toc_order) if e is not None and not pd.isna(e.toc_order) else None,
                    "placed_by": placed[k] if k < len(placed) else "",
                    "shared_span": len(csv_ids) > 1,
                })
        elif r.kind == "toc" and r.entry_id:
            items.append({**common, "key": f"entry:{r.entry_id}", "kind": "derived", "title": r.title, "placed_by": "derived"})
        elif r.gm_ids:
            items.append({**common, "key": f"gm:{_ids(r.gm_ids)[0]}", "kind": "gm", "title": r.title, "placed_by": "gm"})
        elif r.kind == "case":
            items.append({**common, "key": f"case:{r.court_case}", "kind": "case", "title": r.title, "placed_by": "case"})
        else:
            label = "Subdocument" if r.kind == "subdoc" else "Unindexed document"
            items.append({**common, "key": f"{r.kind}:{r.start_scan}", "kind": r.kind, "title": r.title or label,
                          "placed_by": "model"})
    return items


def insert_unplaced(items: list[dict], toc: pd.DataFrame) -> list[dict]:
    """Add the index entries without a segment after the placed entry that precedes them in the index."""
    placed = {it["csv_id"] for it in items if it.get("csv_id")}
    out = list(items)
    for e in toc.itertuples():
        if pd.isna(e.csv_id) or str(int(e.csv_id)) in placed:
            continue
        cid = str(int(e.csv_id))
        order = e.toc_order if not pd.isna(e.toc_order) else float("inf")
        at = 0
        for k, it in enumerate(out):
            if it.get("toc_order") is not None and it["toc_order"] <= order:
                at = k + 1
        # behind the subdocuments of the preceding entry
        while at < len(out) and out[at]["depth"] > 0:
            at += 1
        out.insert(at, {
            "key": f"toc:{cid}", "kind": "unplaced", "csv_id": cid, "depth": 0, "title": e.title,
            "index_folios": _folios(e.folio_start, e.folio_end), "toc_order": float(order) if order != float("inf") else None,
            "start": None, "end": None, "model_start": None, "model_end": None, "placed_by": "not placed",
            "date": "" if pd.isna(e.date_begin) else str(e.date_begin),
        })
    return out


def attach_reviews(items: list[dict], reviews: dict[str, dict], position: dict[str, int]) -> list[dict]:
    """Current review per item; reviewed keys absent from the model output become items of their own."""
    keys = {it["key"] for it in items}
    for it in items:
        rv = reviews.get(it["key"])
        it["review"] = rv
        if rv and rv.get("start_scan"):
            it["start"] = rv["start_scan"]
        if rv and rv.get("end_scan"):
            it["end"] = rv["end_scan"]
    for key, rv in reviews.items():
        if key in keys or not (rv.get("start_scan") or rv.get("model_start")):
            continue
        anchor = position.get(rv.get("start_scan") or rv.get("model_start"), 0)
        at = next((k for k, it in enumerate(items) if it.get("start") and position.get(it["start"], 0) > anchor), len(items))
        added = key.startswith("added:")
        items.insert(at, {
            "key": key, "kind": "added" if added else rv.get("kind") or "stale", "stale": not added, "depth": 0,
            "title": rv.get("title") or ("Added document" if added else "(no longer in the model output)"),
            "start": rv.get("start_scan"), "end": rv.get("end_scan"),
            "model_start": rv.get("model_start"), "model_end": rv.get("model_end"),
            "placed_by": rv.get("placed_by") or ("added" if added else ""), "review": rv,
        })
    return items


def current_reviews(conn, inventory_number: str | None = None) -> dict[tuple[str, str], dict]:
    """Latest review row per (inventory, item_key); cleared reviews (all empty) are left out."""
    where, params = ("WHERE inventory = ?", (inventory_number,)) if inventory_number else ("", ())
    try:
        cur = conn.execute(
            f"SELECT * FROM segment_review WHERE id IN (SELECT max(id) FROM segment_review {where} GROUP BY inventory, item_key)",
            params,
        )
    except Exception:  # table not created yet
        return {}
    cols = [c[0] for c in cur.description]
    out = {}
    for row in cur.fetchall():
        rv = dict(zip(cols, row))
        if rv["start_status"] or rv["end_status"] or rv["verdict"] or rv["note"]:
            out[(rv["inventory"], rv["item_key"])] = rv
    return out


def load_review(conn, inventory_number: str) -> dict:
    """Everything the review page needs for one inventory."""
    from .inventory import load_toc

    inv_id, scans = load_scans(conn, inventory_number)
    position = {s["f"]: k for k, s in enumerate(scans)}
    seg = pd.read_parquet(os.path.join(SEGMENT_DIR, f"{inventory_number}.parquet"))
    seg = seg.assign(segment=pd.to_numeric(seg["segment"])).sort_values("segment")
    toc = load_toc(conn, inv_id)
    items = insert_unplaced(model_items(seg, toc), toc)
    reviews = {k: v for (_, k), v in current_reviews(conn, inventory_number).items()}
    items = attach_reviews(items, reviews, position)
    for it in items:
        it.setdefault("review", None)
    return {"inventory": inventory_number, "scans": scans, "items": items, "n_index": int(toc["csv_id"].notna().sum())}


# ── saving ────────────────────────────────────────────────────────────────────


def _status(model, value, side):
    if not value:
        return None
    return "confirmed" if value == model and not side else "corrected"


def build_rows(data: dict, payload: dict, now: str | None = None) -> list[dict]:
    """The segment_review rows for one save: the item itself and, with
    payload['link'], the neighbours whose shared/adjacent boundary moves along.
    `data` is load_review() output."""
    now = now or datetime.now().isoformat(timespec="seconds")
    items = {it["key"]: it for it in data["items"]}
    position = {s["f"]: k for k, s in enumerate(data["scans"])}
    scans = data["scans"]
    key = payload["key"]
    if key == "added:new":
        n = 1 + max([int(k.split(":")[1]) for k in items if re.fullmatch(r"added:\d+", k)] or [0])
        key = f"added:{n}"
        it = {"key": key, "kind": "added", "placed_by": "added", "model_start": None, "model_end": None, "title": payload.get("title")}
    else:
        it = items[key]
    verdict = payload.get("verdict") or None
    if verdict and verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict}")
    clear = payload.get("clear")
    start = None if clear else payload.get("start") or None
    end = None if clear else payload.get("end") or None
    for s in (start, end):
        if s and s not in position:
            raise ValueError(f"unknown scan {s}")
    if start and end and position[end] < position[start]:
        raise ValueError("the end lies before the start")
    sides = {k: (payload.get(k) if payload.get(k) in SIDES else None) for k in ("start_side", "end_side")}
    common = {"inventory": data["inventory"], "reviewer": payload.get("reviewer") or None, "created_at": now}

    def row(item, **kw):
        return {**common, "item_key": item["key"], "kind": item["kind"], "placed_by": item.get("placed_by"),
                "model_start": item.get("model_start"), "model_end": item.get("model_end"),
                "title": item.get("title") if item["kind"] == "added" else None, **kw}

    # unsure: the scans shown are kept, but not counted as checked
    checked = verdict != "unsure"
    main = row(
        it, start_scan=start, end_scan=end, **sides,
        start_status=_status(it.get("model_start"), start, sides["start_side"]) if checked else None,
        end_status=_status(it.get("model_end"), end, sides["end_side"]) if checked else None,
        verdict=None if clear else verdict, note=None if clear else (payload.get("note") or None),
    )
    rows = [main]
    if clear or not checked or not payload.get("link"):
        return rows

    def human(rv, field):
        return rv is not None and rv.get(field) in ("confirmed", "corrected")

    # Only the neighbouring item in the ToC moves along, and only when the
    # model put their boundaries together: the previous item ending where this
    # one starts (shared) or on the scan before (adjacent), likewise the next.
    # Index entries placed on the same span as this one are left alone.
    placed = [x for x in data["items"] if x["kind"] not in ("gap", "unplaced") and x.get("model_start")]
    k = next((i for i, x in enumerate(placed) if x["key"] == it["key"]), None)
    if k is None:
        return rows
    prev = placed[k - 1] if k > 0 else None
    nxt = placed[k + 1] if k + 1 < len(placed) else None

    def free(nb, field):
        rv = nb.get("review")
        return not (rv and rv.get(f"{field}_status") in ("confirmed", "corrected")) and not (
            nb["model_start"] == it["model_start"] and nb["model_end"] == it["model_end"])

    if prev and start and start != it["model_start"] and it.get("boundary") in ("shared", "adjacent") and free(prev, "end"):
        gap = 0 if it["boundary"] == "shared" else 1
        if position[prev["model_end"]] == position[it["model_start"]] - gap:
            new_end = position[start] - gap
            if new_end >= position[prev.get("start") or prev["model_start"]]:
                rows.append(_implied(row, prev, "end", scans[new_end]["f"]))
    if nxt and end and end != it["model_end"] and nxt.get("boundary") in ("shared", "adjacent") and free(nxt, "start"):
        gap = 0 if nxt["boundary"] == "shared" else 1
        if position[nxt["model_start"]] == position[it["model_end"]] + gap:
            new_start = position[end] + gap
            if new_start < len(scans) and new_start <= position[nxt.get("end") or nxt["model_end"]]:
                rows.append(_implied(row, nxt, "start", scans[new_start]["f"]))
    return rows


def _implied(row, nb, field, scan):
    """A neighbour's review with one boundary moved along; its other labels are kept."""
    rv = nb.get("review") or {}
    kept = {k: rv.get(k) for k in ("start_scan", "end_scan", "start_side", "end_side", "start_status", "end_status", "verdict", "note")}
    kept[f"{field}_scan"] = scan
    kept[f"{field}_side"] = None
    kept[f"{field}_status"] = "implied"
    return row(nb, **kept)


# ── evaluation ────────────────────────────────────────────────────────────────

BUCKETS = ["exact", "±1", "±2–5", ">5"]


def _bucket(model, reviewed):
    a, b = scan_number(model), scan_number(reviewed)
    if a is None or b is None:
        return None
    d = abs(a - b)
    return "exact" if d == 0 else "±1" if d == 1 else "±2–5" if d <= 5 else ">5"


def evaluate(reviews: list[dict]) -> pd.DataFrame:
    """Per placement method: reviewed boundaries by distance between the model's
    and the reviewed scan, plus the document-level verdicts. Implied boundaries
    are not counted."""
    rows = []
    for rv in reviews:
        method = (rv.get("placed_by") or rv.get("kind") or "").split(";")[0] or "?"
        for field in ("start", "end"):
            if rv.get(f"{field}_status") in ("confirmed", "corrected") and rv.get(f"model_{field}"):
                rows.append({"method": method, "what": field, "bucket": _bucket(rv[f"model_{field}"], rv[f"{field}_scan"])})
        if rv.get("verdict"):
            rows.append({"method": method, "what": "verdict", "bucket": rv["verdict"]})
        if rv.get("kind") == "unplaced" and rv.get("start_status"):
            rows.append({"method": method, "what": "verdict", "bucket": "found"})
        if rv.get("kind") == "added":
            rows.append({"method": "added", "what": "verdict", "bucket": "missed by model"})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["col"] = df["what"].where(df["what"] == "verdict", df["what"] + " " + df["bucket"].astype(str))
    df.loc[df["what"] == "verdict", "col"] = df["bucket"]
    table = df.pivot_table(index="method", columns="col", aggfunc="size", fill_value=0)
    order = [f"{w} {b}" for w in ("start", "end") for b in BUCKETS]
    cols = [c for c in order if c in table.columns] + sorted(c for c in table.columns if c not in order)
    table = table[cols]
    table.loc["all"] = table.sum()
    return table


def export_rows(conn) -> pd.DataFrame:
    return pd.DataFrame(list(current_reviews(conn).values()))
