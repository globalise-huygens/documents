#!/usr/bin/env python3
"""
Step 23: date the index entries from their descriptions.

The enriched index (step 7, begin_of_begin ... end_of_end) dates about 146,000
of the 226,000 entries of the digitized indexes; the typoscripts themselves
only give years (YEAR (EARLIEST), YEAR (LATEST)). This reads the dates in the
description (segmentation/title_dates.py): the corrected text of OBP
NT_gecorrigeerd (INHOUD) where the entry has one, else the title.
date_earliest_begin and date_latest_begin are the first date (its first and
last day: "Maart 1783" is 1 to 31 March), date_earliest_end and
date_latest_end the last.

  - Entries without any date get the dates of the description. One whose only
    date is that of receipt ("Register der papieren (ontfangen 5 October 1690
    per d'Gideon)") gets only an end, as in the enriched index; one without
    dates in the text those of NT's JAAR column.
  - Descriptions with a day and month but no year ("dato 14 September") get
    the typoscript's year or the year before (title_dates.guess_year, from the
    receipt date or the dated neighbouring entries). This replaces the dates the
    enriched index gave about 2,000 of them, whose years do not come from the
    text (often decades off), and gives an entry with only a receipt date its
    own date.
  - Other existing dates are not changed.

Then gives the segmentation documents of the changed entries the same dates,
where they had the entry's previous ones (title from the index). Later imports
copy them from the entries (segmentation/db_import.py), so the segmentation
need not be run again. Running it again changes nothing.

  uv run python 23_add_title_dates.py [--dry-run]
"""

import argparse
import io
import os
import zipfile
from collections import Counter

import pandas as pd
from sqlalchemy import create_engine, text

from segmentation.title_dates import date_range, guess_year, nt_dates, receipt_dates, title_dates, yearless_dates, yearless_range

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NT_XLSX = os.path.join(DATA, "OBP NT_gecorrigeerd.xlsx")
INDEX_ZIP = os.path.join(DATA, "GLOBALISE - Digitized Indexes of the Dutch East India Company OBP (1602-1799).csv.zip")
DATE_COLUMNS = ("date_earliest_begin", "date_latest_begin", "date_earliest_end", "date_latest_end")
NEIGHBOURS = 3  # entries to look for a dated neighbour, before and after


def typoscript_years() -> dict[str, int]:
    """OBP id -> YEAR (EARLIEST) of the digitized indexes."""
    with zipfile.ZipFile(INDEX_ZIP) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        t = pd.read_csv(io.BytesIO(z.read(name)), usecols=["ID", "YEAR (EARLIEST)"], low_memory=False)
    t = t.dropna()
    return dict(zip(t["ID"].astype(str), t["YEAR (EARLIEST)"].astype(int)))


def _iso(r):
    return {c: d and d.isoformat() for c, d in zip(DATE_COLUMNS, r)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="count only, change nothing")
    args = ap.parse_args()
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        entries = pd.read_sql(text(
            "SELECT d.id, i.inventory_number AS inv, d.toc_order, d.title, d.nt_index, " + ", ".join(f"d.{c}" for c in DATE_COLUMNS) + ", "
            "(SELECT min(e.identifier) FROM document2external_id de JOIN external_id e ON e.id = de.external_id AND e.context = 'OBP_INDEX' "
            " WHERE de.document_id = d.id) AS obp "
            "FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' "
            "JOIN inventory i ON i.id = d.inventory_id"), conn)
        entries = entries.sort_values(["inv", "toc_order", "obp"]).reset_index(drop=True)
        nt = pd.read_excel(NT_XLSX, usecols=["ID", "INHOUD", "JAAR"], dtype=str).dropna(subset=["ID"])
        nt = nt.set_index(nt["ID"].astype(int))
        years = typoscript_years()

        def nt_row(e):
            return nt.loc[int(e.nt_index)] if pd.notna(e.nt_index) and int(e.nt_index) in nt.index else None

        bodies = [n["INHOUD"] if (n := nt_row(e)) is not None and isinstance(n["INHOUD"], str) else e.title for e in entries.itertuples()]
        own = [title_dates(b) for b in bodies]
        receipts = [receipt_dates(b) for b in bodies]
        # the first date of each description with a year, for the neighbours of yearless ones
        anchors = [(o or r or [None])[0] for o, r in zip(own, receipts)]
        inv = entries["inv"].tolist()

        def neighbours(k):
            out = []
            for step in (-1, 1):
                j = k + step
                while 0 <= j < len(inv) and inv[j] == inv[k] and abs(j - k) <= NEIGHBOURS:
                    if anchors[j] is not None:
                        out.append(anchors[j])
                        break
                    j += step
            return out

        rows, how = [], Counter()
        for k, e in enumerate(entries.itertuples()):
            old = tuple(getattr(e, c) for c in DATE_COLUMNS)
            undated = all(v is None for v in old)
            r = None
            if not own[k] and (items := yearless_dates(bodies[k])) and e.obp in years:
                # a day and month without a year: replaces the dates the enriched index made up, or a receipt-only end
                r = yearless_range(items, guess_year(items, years[e.obp], (receipts[k] or [None])[0], neighbours(k)))
                if r is not None:
                    how["year of the typoscript or the year before" + (" (replacing dates)" if old[0] is not None else "")] += 1
            if r is None and undated:
                if (r := date_range(own[k])) is not None:
                    how["description"] += 1
                elif (r := date_range(receipts[k])) is not None:
                    how["only an end (date of receipt)"] += 1
                    r = (None, None, r[2], r[3])  # written before it was received
                elif (n := nt_row(e)) is not None and (r := date_range(nt_dates(n["JAAR"]))) is not None:
                    how["NT's JAAR"] += 1
            if r is not None and _iso(r) != dict(zip(DATE_COLUMNS, old)):
                rows.append({"id": e.id, **_iso(r), **{f"old_{c}": v for c, v in zip(DATE_COLUMNS, old)}})
        print(f"index entries: {len(entries):,}; without dates: {sum(all(getattr(e, c) is None for c in DATE_COLUMNS) for e in entries.itertuples()):,}")
        for h, n in how.most_common():
            print(f"  {h}: {n:,}")
        print(f"entries whose dates change: {len(rows):,}")
        if args.dry_run:
            titles = entries.set_index("id")["title"]
            for r in rows[:: max(1, len(rows) // 20)]:
                print(f"  {r['date_earliest_begin'] or '…':10} – {r['date_latest_end']}  (was {r['old_date_earliest_begin'] or '-'})  {str(titles[r['id']])[:100]}")
            return
        sets = ", ".join(f"{c} = :{c}" for c in DATE_COLUMNS)
        conn.execute(text(f"UPDATE document SET {sets} WHERE id = :id"), rows)
        # segmentation documents of these entries (they share the entry's OBP_INDEX ExternalID) that have its previous dates
        same = " AND ".join(f"d.{c} IS :old_{c}" for c in DATE_COLUMNS)
        n = conn.execute(text(
            f"UPDATE document AS d SET {sets} WHERE d.title_source = 'VOC' AND {same} "
            "AND d.method_id = (SELECT id FROM document_identification_method WHERE name = 'Segmentation model') "
            "AND d.id IN (SELECT sd.document_id FROM document2external_id ie "
            "JOIN external_id e ON e.id = ie.external_id AND e.context = 'OBP_INDEX' "
            "JOIN document2external_id sd ON sd.external_id = e.id AND sd.document_id != ie.document_id WHERE ie.document_id = :id)"),
            rows).rowcount
        print(f"index entries changed: {len(rows):,}; segmentation documents given their dates: {n:,}")


if __name__ == "__main__":
    main()
