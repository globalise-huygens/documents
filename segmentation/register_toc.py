"""
A volume's handwritten registers (register_entries.csv, rows.py) as its ToC.

The entries go to the same ToC alignment as the OBP indexes (segmenter.py):
folio numbers where the entry has them and the volume shows them, otherwise
the entry's words against the opening words of the scans, header dates
against the entry's date, and the start evidence. Order: by folio when most
entries have one (a register lists the papers by kind, not in the order they
are bound), else in register order (item numbers follow the binding).

Placed entries become documents with the register's title (dittos resolved),
part of no index: their evidence records the register (first scan), the entry,
its folio range or item number and how it was placed.
"""

import os

import numpy as np
import pandas as pd

from .header_dates import parse_header_date
from .segmenter import segment_inventory

REGISTER_ENTRIES = "register_entries.csv"
MIN_TITLE_WORDS = 3


def load_register_entries(path: str = REGISTER_ENTRIES) -> dict[str, pd.DataFrame]:
    if not path or not os.path.exists(path):
        return {}
    e = pd.read_csv(path, dtype={"inventory": str, "item": str}, low_memory=False)
    return {inv: g.reset_index(drop=True) for inv, g in e.groupby("inventory")}


def _date(text) -> object:
    if not isinstance(text, str):
        return None
    d = parse_header_date(text)
    return d.date if d is not None and d.precision != "year" else None


def entries_to_toc(entries: pd.DataFrame, register_scans: set[str]) -> pd.DataFrame:
    """The entries in the shape of InventoryData.toc (no index ids)."""
    e = entries[entries["title"].fillna("").str.split().str.len() >= MIN_TITLE_WORDS].copy()
    e = e.reset_index(drop=True)
    by_folio = e["folio_start"].notna().mean() >= 0.5
    e["order_key"] = np.where(e["folio_start"].notna(), e["folio_start"], np.nan) if by_folio else np.arange(len(e))
    if by_folio:
        e = e[e["folio_start"].notna()]  # unnumbered entries cannot be ordered among the numbered ones
    e = e.sort_values(["order_key", "register_first_scan", "entry"]).reset_index(drop=True)
    dates = e["date"].map(_date)
    toc = pd.DataFrame({
        "doc_id": [f"register:{r.register_first_scan}:{r.entry}" for r in e.itertuples()],
        "csv_id": np.nan,
        "title": e["title"],
        "folio_start": e["folio_start"].astype(float),
        "folio_end": e["folio_end"].astype(float),
        "start_side": None, "end_side": None,
        "toc_order": np.arange(len(e), dtype=float),
        "folio_sequence": 1.0,
        "katern": None,
        "date_begin": dates, "date_end": dates,
        "register_first_scan": e["register_first_scan"], "entry": e["entry"], "item": e["item"],
    })
    toc["end_eff"] = toc["folio_end"].where(toc["folio_end"] >= toc["folio_start"], toc["folio_start"])
    return toc


def segment_with_register(inv, model, entries: pd.DataFrame):
    """Segment an inventory with its register entries as ToC; the register pages are not documents."""
    register_scans = set()
    for first, last in entries[["register_first_scan", "register_last_scan"]].drop_duplicates().itertuples(index=False):
        prefix, a, b = first.rsplit("_", 1)[0], int(first.rsplit("_", 1)[1]), int(last.rsplit("_", 1)[1])
        register_scans |= {f"{prefix}_{k:04d}" for k in range(a, b + 1)}
    inv.toc = entries_to_toc(entries, register_scans)
    inv.features, inv.toc_text_sim = None, None
    res = segment_inventory(inv, model, use_toc=True)
    how = {pl.entry: pl.how for pl in res.placements}
    fn = inv.scans["filename"]
    for sg in res.segments:
        if sg.kind == "toc" and sg.toc_rows:
            r = inv.toc.loc[sg.toc_rows[0]]
            parent = inv.toc.at[sg.parent_row, "doc_id"] if sg.parent_row is not None else None
            sg.derived = {
                "entry_id": r.doc_id, "parent_entry_id": parent, "title": r.title,
                "date_begin": r.date_begin.isoformat() if r.date_begin else None, "date_end": r.date_end.isoformat() if r.date_end else None,
                "register": {"first_scan": r.register_first_scan, "entry": int(r.entry), "item": r.item if isinstance(r.item, str) else None,
                             "folio_start": None if pd.isna(r.folio_start) else int(r.folio_start),
                             "folio_end": None if pd.isna(r.folio_end) else int(r.folio_end), "placed_by": how.get(sg.toc_rows[0])},
                "on_register_page": fn.iat[sg.start] in register_scans,
            }
        elif sg.kind == "subdoc" and sg.parent_row is not None:
            sg.derived = {"parent_entry_id": inv.toc.at[sg.parent_row, "doc_id"]}
    return res
