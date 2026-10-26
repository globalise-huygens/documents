#!/usr/bin/env python3
"""
Step 20: record who made each document title (document.title_source).

  VOC                   the index entry (TANAP / typoscript): the VOC itself named the document
  VOC (other version)   the index entry of another version of the same text (derived ToC, versions.py)
  Resilient Diversity   the court case dataset (data/EMDCCR*.xlsx)
  General Missives      the General Missives identified by hand
  page header           the first running header of a baseline document (step 16)
  inferred              made by this project from the text (set when inferred titles are imported)

Adds the column if missing and fills it for the existing documents; the
segmentation import (segmentation/db_import.py) sets it for the documents it
creates. Segmentation documents get their source from their stored evidence.

  uv run python 20_add_title_source.py [--dry-run]
"""

import argparse
import os

from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")

BY_METHOD = {
    "TANAP Digitized Index": "VOC",
    "General Missives Ground Truth": "General Missives",
    "Baseline: Empty Pages & Signatures": "page header",
}
SEGMENTATION = "Segmentation model"
# segmentation documents: the first matching key of their evidence decides
BY_EVIDENCE = (
    ("$.court_case", "Resilient Diversity"),
    ("$.derived_entry", "VOC (other version)"),
    ("$.general_missives", "General Missives"),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="count only, change nothing")
    args = ap.parse_args()
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        if "title_source" not in {c["name"] for c in inspect(conn).get_columns("document")}:
            if args.dry_run:
                print("would add column document.title_source")
                return
            conn.execute(text("ALTER TABLE document ADD COLUMN title_source VARCHAR(32)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_title_source ON document (title_source)"))
        method = dict(conn.execute(text("SELECT name, id FROM document_identification_method")).fetchall())
        updates = []
        for name, source in BY_METHOD.items():
            if name in method:
                updates.append((f"{name} → {source}", "UPDATE document SET title_source = :s WHERE method_id = :m AND title IS NOT NULL", {"s": source, "m": method[name]}))
        if SEGMENTATION in method:
            m = method[SEGMENTATION]
            for path, source in BY_EVIDENCE:
                updates.append((f"{SEGMENTATION}, evidence {path} → {source}",
                                "UPDATE document SET title_source = :s WHERE method_id = :m AND title IS NOT NULL AND title_source IS NULL "
                                "AND id IN (SELECT document_id FROM document_evidence WHERE json_extract(evidence, :p) IS NOT NULL)",
                                {"s": source, "m": m, "p": path}))
            updates.append((f"{SEGMENTATION}, other titled documents (ToC entries) → VOC",
                            "UPDATE document SET title_source = 'VOC' WHERE method_id = :m AND title IS NOT NULL AND title_source IS NULL", {"m": m}))
        for label, sql, params in updates:
            if args.dry_run:
                count_sql = "SELECT count(*) FROM document WHERE " + sql.split(" WHERE ", 1)[1]
                print(f"{label}: {conn.execute(text(count_sql), params).scalar():,}")
            else:
                print(f"{label}: {conn.execute(text(sql), params).rowcount:,}")
        if args.dry_run:
            raise SystemExit(0)
        for source, n in conn.execute(text("SELECT title_source, count(*) FROM document GROUP BY 1 ORDER BY 2 DESC")):
            print(f"  {source or '(no title)'}: {n:,}")


if __name__ == "__main__":
    main()
