"""
Load one inventory from the database: its scans in physical order with their
raw page metadata, its ToC entries, and (for evaluation) ground truth.

Scans are the unit of segmentation. A Double scan holds two pages (the verso
of one leaf and the recto of the next); both Page rows carry the same
metadata, so features are per scan. Positions are 0..N-1 in (scan_order,
filename) order; scan_order itself is not contiguous.
"""

import ast
import os
import re
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
TANAP_METHOD = "TANAP Digitized Index"
GM_METHOD = "General Missives Ground Truth"
BASELINE_METHOD = "Baseline: Empty Pages & Signatures"

# ToC entries whose document is not in the volume: "[ontbreekt]", or "[ontbreekt"
# with an explanation ("[ontbreekt maar geinsereert bij resolutie van ...]",
# "[ontbreekt; zie VOC 08068 ...]"). Not "[folio 603 en 604 ontbreken]" and the
# like: there only part of the document is missing.
MISSING_RE = r"\[ontbreekt"

BLANK_CHARS = 20  # 3.5_import_empty_pages.py: a page with less text is blank

# Validated inventories whose segmentation import is effectively empty
BROKEN_VALIDATIONS = {"1568", "1574", "8165"}


def connect(database_url: str = DATABASE_URL) -> sqlite3.Connection:
    path = database_url.removeprefix("sqlite:///")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _parse_list(raw) -> str:
    """"['a ', 'b']" → "a | b"; plain strings pass through."""
    if raw is None or raw == "" or raw == "[]":
        return ""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return " | ".join(str(p).strip() for p in parsed if str(p).strip())
    except (ValueError, SyntaxError):
        pass
    return str(raw).strip()


def _parse_folios(raw) -> list[int]:
    if not raw:
        return []
    return [int(p) for p in str(raw).split(",") if re.fullmatch(r"\s*\d{1,4}\s*", p)]


@dataclass
class InventoryData:
    inventory_number: str
    inventory_id: str
    scans: pd.DataFrame  # one row per scan, index = position
    toc: pd.DataFrame  # one row per ToC entry, sorted by toc_order (NULL last)
    features: pd.DataFrame = field(default=None)  # filled by predictors
    texts: pd.Series = field(default=None)  # full text per scan (texts.load_texts), loaded on demand
    toc_text_sim: object = field(default=None)  # ToC entries × scans text similarity, filled by predictors

    @property
    def n(self) -> int:
        return len(self.scans)


def drop_missing(toc: pd.DataFrame) -> pd.DataFrame:
    """The ToC entries without those whose document is missing from the volume (MISSING_RE)."""
    return toc[~toc["title"].fillna("").str.contains(MISSING_RE, case=False, regex=True)]


def load_toc(conn: sqlite3.Connection, inventory_id: str) -> pd.DataFrame:
    """The inventory's ToC entries (TANAP / OBP index), sorted by toc_order (NULL last)."""
    toc = pd.read_sql(
        "SELECT d.id AS doc_id, CAST(e.identifier AS INTEGER) AS csv_id, d.title, "
        "       d.folio_start, d.folio_end, d.toc_folio_start_side AS start_side, "
        "       d.toc_folio_end_side AS end_side, d.toc_order, d.toc_folio_sequence AS folio_sequence, "
        "       d.toc_katern AS katern, d.date_earliest_begin AS date_begin, d.date_latest_end AS date_end "
        "FROM document d "
        "JOIN document_identification_method m ON m.id = d.method_id AND m.name = ? "
        "LEFT JOIN document2external_id de ON de.document_id = d.id "
        "LEFT JOIN external_id e ON e.id = de.external_id AND e.context = 'OBP_INDEX' "
        "WHERE d.inventory_id = ? AND (e.id IS NOT NULL OR de.id IS NULL)",
        conn,
        params=(TANAP_METHOD, inventory_id),
    )
    toc = drop_missing(toc.drop_duplicates("doc_id"))
    # an all-NULL column comes back as object/None; keep these numeric (NaN)
    for c in ("csv_id", "folio_start", "folio_end", "toc_order", "folio_sequence"):
        toc[c] = pd.to_numeric(toc[c], errors="coerce")
    for c in ("date_begin", "date_end"):
        toc[c] = pd.to_datetime(toc[c], errors="coerce").dt.date
    toc["end_eff"] = toc["folio_end"].where(toc["folio_end"] >= toc["folio_start"], toc["folio_start"])
    return toc.sort_values(["toc_order", "csv_id"], na_position="last").reset_index(drop=True)


def load_inventory(conn: sqlite3.Connection, inventory_number: str) -> InventoryData:
    inv = conn.execute(
        "SELECT id FROM inventory WHERE inventory_number = ?", (inventory_number,)
    ).fetchone()
    if inv is None:
        raise KeyError(f"inventory {inventory_number} not found")
    inv_id = inv[0]

    scans = pd.read_sql(
        "SELECT id AS scan_id, filename, scan_order, scan_type, languages, "
        "       inventory_text_end_offset - inventory_text_start_offset AS text_len "
        "FROM scan WHERE inventory_id = ? ORDER BY scan_order, filename",
        conn,
        params=(inv_id,),
    )
    pages = pd.read_sql(
        "SELECT scan_id, page_or_folio_number, header, signatures, is_blank, has_marginalia "
        "FROM page WHERE inventory_id = ?",
        conn,
        params=(inv_id,),
    )
    agg = pages.groupby("scan_id").agg(
        n_pages=("scan_id", "size"),
        is_blank=("is_blank", lambda s: bool(s.fillna(1).astype(bool).all())),
        folio_raw=("page_or_folio_number", "first"),
        header_raw=("header", "first"),
        signatures_raw=("signatures", "first"),
        marginalia=("has_marginalia", lambda s: bool(s.fillna(0).astype(bool).any())),
    )
    scans = scans.merge(agg, left_on="scan_id", right_index=True, how="left")
    scans["n_pages"] = scans["n_pages"].fillna(0).astype(int)
    scans["marginalia"] = scans["marginalia"].fillna(False).astype(bool)
    scans["layout"] = scans["scan_type"].map({"Single": "single", "Double": "double"}).fillna("single")
    scans["header"] = scans["header_raw"].map(_parse_list)
    scans["signatures"] = scans["signatures_raw"].map(_parse_list)
    scans["folios"] = scans["folio_raw"].map(_parse_folios)
    scans["languages"] = scans["languages"].fillna("")
    scans["text_len"] = scans["text_len"].fillna(0).astype(int)
    # no page rows (20 inventories lack them): blank by the length of the scan's text, as in step 3.5
    scans["is_blank"] = scans["is_blank"].where(scans["n_pages"] > 0, scans["text_len"] < BLANK_CHARS).astype(bool)
    scans = scans.drop(columns=["header_raw", "signatures_raw", "folio_raw"]).reset_index(drop=True)

    toc = load_toc(conn, inv_id)

    return InventoryData(inventory_number, inv_id, scans, toc)


# ── ground truth ──────────────────────────────────────────────────────────────


def _doc_spans(conn, inv: InventoryData, where: str, params: tuple) -> pd.DataFrame:
    """Documents with the first/last scan position of their linked pages."""
    pos = pd.Series(inv.scans.index, index=inv.scans["scan_id"])
    links = pd.read_sql(
        "SELECT d.id AS doc_id, d.part_of_id, p.scan_id "
        "FROM page2document p2d "
        "JOIN document d ON d.id = p2d.document_id "
        "JOIN document_identification_method m ON m.id = d.method_id "
        "JOIN page p ON p.id = p2d.page_id "
        f"WHERE d.inventory_id = ? AND {where}",
        conn,
        params=(inv.inventory_id, *params),
    )
    if links.empty:
        return pd.DataFrame(columns=["doc_id", "part_of_id", "start", "end", "csv_id"])
    links["pos"] = links["scan_id"].map(pos)
    spans = (
        links.dropna(subset=["pos"])
        .groupby(["doc_id"], dropna=False)
        .agg(part_of_id=("part_of_id", "first"), start=("pos", "min"), end=("pos", "max"))
        .reset_index()
    )
    ids = pd.read_sql(
        "SELECT de.document_id AS doc_id, CAST(e.identifier AS INTEGER) AS csv_id "
        "FROM document2external_id de JOIN external_id e ON e.id = de.external_id "
        "WHERE e.context = 'OBP_INDEX' AND de.document_id IN (SELECT id FROM document WHERE inventory_id = ?)",
        conn,
        params=(inv.inventory_id,),
    ).drop_duplicates("doc_id")
    spans = spans.merge(ids, on="doc_id", how="left")
    spans[["start", "end"]] = spans[["start", "end"]].astype(int)
    return spans.sort_values(["start", "end"]).reset_index(drop=True)


def load_validated(conn, inv: InventoryData) -> pd.DataFrame:
    """Manually validated documents and subdocuments (step 15)."""
    return _doc_spans(conn, inv, "p2d.source = 'SEGMENTATION'", ())


def load_general_missives(conn, inv: InventoryData) -> pd.DataFrame:
    """Hand-checked General Missives (step 8). Their ToC entry may extend
    beyond the span (appendices), so only starts and interiors are reliable."""
    return _doc_spans(conn, inv, "m.name = ?", (GM_METHOD,))


def load_gm_overrides(conn, inv: InventoryData) -> pd.DataFrame:
    """The hand-checked General Missives of the inventory as overrides for
    segment_inventory: gm_id (their document), start, end (scan positions),
    csv_id (index id of their ToC entry), title, date_begin, date_end."""
    spans = load_general_missives(conn, inv)
    if spans.empty:
        return pd.DataFrame(columns=["gm_id", "start", "end", "csv_id", "title", "date_begin", "date_end"])
    info = pd.read_sql(
        "SELECT d.id AS doc_id, d.title, d.date_earliest_begin AS date_begin, coalesce(d.date_latest_end, d.date_latest_begin) AS date_end "
        "FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = ? WHERE d.inventory_id = ?",
        conn,
        params=(GM_METHOD, inv.inventory_id),
    )
    out = spans.merge(info, on="doc_id").rename(columns={"doc_id": "gm_id"})
    return out[["gm_id", "start", "end", "csv_id", "title", "date_begin", "date_end"]]


def load_baseline(conn, inv: InventoryData) -> pd.DataFrame:
    return _doc_spans(conn, inv, "m.name = ?", (BASELINE_METHOD,))


def validated_inventories(conn) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT i.inventory_number FROM page2document p2d "
        "JOIN document d ON d.id = p2d.document_id JOIN inventory i ON i.id = d.inventory_id "
        "WHERE p2d.source = 'SEGMENTATION'"
    ).fetchall()
    return sorted(r[0] for r in rows if r[0] not in BROKEN_VALIDATIONS)


def gm_inventories(conn) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT i.inventory_number FROM document d "
        "JOIN document_identification_method m ON m.id = d.method_id AND m.name = ? "
        "JOIN inventory i ON i.id = d.inventory_id "
        "WHERE EXISTS (SELECT 1 FROM page2document p WHERE p.document_id = d.id)",
        (GM_METHOD,),
    ).fetchall()
    return sorted(r[0] for r in rows)
