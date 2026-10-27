#!/usr/bin/env python3
"""
Step 22: give documents the GLOB id of their place (document.place_glob_id).

The id is that of the GLOBALISE places dataset
(data/GLOBALISE - Places in the Dutch East India Company Archives (1602-1799).xlsx).
Adds the column if missing and fills it for the documents with a settlement
(location_id) whose GLOB id the places dataset has. The segmentation import
(segmentation/db_import.py) sets it for the documents it creates, also for
places of inferred titles that are not settlements; it never adds settlements.

Prints the settlements whose GLOB id is missing from the places dataset.

  uv run python 22_add_place_glob_id.py [--dry-run]
"""

import argparse
import os

from sqlalchemy import create_engine, inspect, text

from segmentation.doc_titles import Places

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="count only, change nothing")
    args = ap.parse_args()
    known = Places().ids
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        settlements = conn.execute(text(
            "SELECT s.id, s.glob_id, (SELECT min(label) FROM settlement_label l WHERE l.settlement_id = s.id) FROM settlement s")).fetchall()
        for _, glob_id, label in settlements:
            if glob_id not in known:
                print(f"not in the places dataset: settlement {label} ({glob_id})")
        has_column = "place_glob_id" in {c["name"] for c in inspect(conn).get_columns("document")}
        if args.dry_run:
            if not has_column:
                print("would add column document.place_glob_id")
            n = conn.execute(text("SELECT count(*) FROM document d JOIN settlement s ON s.id = d.location_id")).scalar()
            print(f"documents with a settlement: {n:,}")
            return
        if not has_column:
            conn.execute(text("ALTER TABLE document ADD COLUMN place_glob_id VARCHAR(32)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_place_glob_id ON document (place_glob_id)"))
        n = 0
        for sid, glob_id, _ in settlements:
            if glob_id in known:
                n += conn.execute(text("UPDATE document SET place_glob_id = :g WHERE location_id = :s AND place_glob_id IS NULL"),
                                  {"g": glob_id, "s": sid}).rowcount
        print(f"documents given the GLOB id of their settlement: {n:,}")
        print(f"documents with a place GLOB id: {conn.execute(text('SELECT count(*) FROM document WHERE place_glob_id IS NOT NULL')).scalar():,}")


if __name__ == "__main__":
    main()
