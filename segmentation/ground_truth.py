"""
Validated ground truth read straight from the "<inv> - Document Segmentation.csv"
files in data/ (the files behind step 15).

Each row is a scan; a scan appears on two rows when one document ends on it
and the next starts on it. Columns:
  TANAP Boundaries           START / END (or "END/START" in one cell)
  TANAP ID                   id of the ToC document the scan belongs to
  Subdocument boundaries     "/"-separated START/END events of subdocuments
  Type of non-document page  Cover / Empty / Document title page / ... for
                             scans that belong to no document

Returns document spans in the same shape as inventory.load_validated() plus
the non-document page type per scan position.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from .inventory import InventoryData

CSV_GLOB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "* - Document Segmentation.csv")
REQUIRED = ["Scan File_Name", "TANAP Boundaries", "TANAP ID", "Subdocument boundaries", "Type of non-document page"]

PAGE_TYPES = {
    "empty": "empty",
    "cover": "cover",
    "table of contents": "table of contents",
    "tables of contents": "table of contents",
    "document title page": "title page",
    "title page": "title page",
    "section title page": "section title page",
    "folio/page number": "number only",
    "small note": "small note",
}


def csv_path(inventory_number: str) -> str | None:
    for p in glob.glob(CSV_GLOB):
        if os.path.basename(p).split(" - ")[0].strip() == inventory_number:
            return p
    return None


def read_csv(path: str) -> pd.DataFrame:
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        if set(REQUIRED) <= set(df.columns):
            return df[REQUIRED].fillna("")
    raise ValueError(f"{path}: expected columns {REQUIRED}")


def _tokens(cell: str) -> list[str]:
    return [t.strip().upper() for t in cell.split("/") if t.strip().upper() in ("START", "END")]


def load_truth(inv: InventoryData, path: str) -> tuple[pd.DataFrame, pd.Series]:
    """(document spans, non-document page type per scan position or '')."""
    df = read_csv(path)
    pos_of = {f: i for i, f in enumerate(inv.scans["filename"])}
    df["pos"] = df["Scan File_Name"].str.strip().map(pos_of)
    missing = df["pos"].isna().sum()
    if missing:
        df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int)

    # top-level (ToC) documents: every row carrying the id. A row on which one
    # document ends and the next starts has "END/START" and "id1/id2"; dots are
    # thousands separators ("40.600"). SAME AS rows are reproductions.
    rows = df[~df["TANAP Boundaries"].str.upper().str.startswith("SAME AS")].copy()
    rows["csv_id"] = rows["TANAP ID"].str.replace(".", "", regex=False).str.split("/")
    rows = rows.explode("csv_id")
    rows["csv_id"] = rows["csv_id"].str.strip()
    rows = rows[rows["csv_id"].str.fullmatch(r"\d+", na=False)]
    top = rows.groupby("csv_id").agg(start=("pos", "min"), end=("pos", "max")).reset_index()
    top["csv_id"] = top["csv_id"].astype(int)
    top["doc_id"] = "toc-" + top["csv_id"].astype(str)
    top["part_of_id"] = None

    # subdocuments: START/END events in row order
    subs, open_at, k = [], None, 0
    for r in df.itertuples():
        for tok in _tokens(r[4]):  # "Subdocument boundaries"
            if tok == "START":
                open_at = r.pos
            elif open_at is not None:
                parent = top[(top["start"] <= open_at) & (top["end"] >= open_at)]
                subs.append(
                    {
                        "doc_id": f"sub-{k}",
                        "part_of_id": parent["doc_id"].iloc[0] if len(parent) else None,
                        "start": open_at,
                        "end": r.pos,
                        "csv_id": np.nan,
                    }
                )
                k += 1
                open_at = None
    spans = pd.concat([top[["doc_id", "part_of_id", "start", "end", "csv_id"]], pd.DataFrame(subs)], ignore_index=True)
    spans = spans.sort_values(["start", "end"]).reset_index(drop=True)

    page_type = pd.Series("", index=inv.scans.index)
    types = df[df["Type of non-document page"].str.strip() != ""]
    for r in types.itertuples():
        label = PAGE_TYPES.get(re.sub(r"\s+", " ", r[5].strip().lower()), "other")
        page_type.iat[r.pos] = label
    return spans, page_type
