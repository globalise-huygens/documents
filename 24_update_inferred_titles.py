#!/usr/bin/env python3
"""
Step 24: give the segmentation documents with an inferred title the titles of a
new run of infer-titles, without importing the segmentation again.

Inferred titles (segmentation/doc_titles.py) come from the text only, so a
change in how they are made (the place as written with its standard name,
"Batavia (Jakarta)"; Manaar is Mannar Island, not Manappād) needs no new
segmentation. The documents have no segment number: a document is found by
its inventory, first scan and the title of the previous run (--old), and gets
the title, place GLOB id and settlement of the new run (--new). Dates and types
are not changed (both runs read them the same way). Documents with more than
one match, or none, are counted and left as they are. Running it again changes
nothing.

  uv run python -m segmentation infer-titles --out inferred_titles.new.csv
  uv run python 24_update_inferred_titles.py --old inferred_titles.csv --new inferred_titles.new.csv [--dry-run]
  mv inferred_titles.new.csv inferred_titles.csv    # what later imports and the review pages use
"""

import argparse
import os
from collections import Counter

import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
KEY = ["inventory", "segment"]


def read(path: str) -> pd.DataFrame:
    t = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return t.set_index(KEY)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="inferred_titles.csv", help="the titles the documents have now")
    ap.add_argument("--new", required=True, help="the titles to give them")
    ap.add_argument("--dry-run", action="store_true", help="count only, change nothing")
    args = ap.parse_args()
    old, new = read(args.old), read(args.new)
    both = old.join(new, how="inner", rsuffix="_new")
    changed = both[(both.title.fillna("") != both.title_new.fillna("")) | (both.place_glob_id.fillna("") != both.place_glob_id_new.fillna(""))]
    changed = changed[changed.title.notna() & changed.title_new.notna()]
    print(f"segments in both runs: {len(both):,}; title or place changed: {len(changed):,}")
    engine = create_engine(DATABASE_URL)
    stats = Counter()
    with engine.begin() as conn:
        settlement = {g: s for g, s in conn.execute(text("SELECT glob_id, id FROM settlement WHERE glob_id IS NOT NULL"))}
        docs = {}  # (inventory, first scan, title) -> [(document, settlement)]
        rows = conn.execute(text(
            "SELECT d.id, i.inventory_number, s.filename, d.title, d.location_id FROM document d "
            "JOIN inventory i ON i.id = d.inventory_id "
            "JOIN page2document pd ON pd.document_id = d.id AND pd.\"index\" = 0 "
            "JOIN page p ON p.id = pd.page_id JOIN scan s ON s.id = p.scan_id "
            "WHERE d.title_source = 'inferred'"))
        for doc_id, inv, scan, title, loc in rows:
            docs.setdefault((inv, scan, title), []).append((doc_id, loc))
        for (inv, _), r in changed.iterrows():
            hits = docs.get((inv, r.start_scan, f"[{r.title}]"), [])
            if len(hits) != 1:
                stats["no document" if not hits else "more than one document"] += 1
                continue
            doc_id, old_loc = hits[0]
            glob = r.place_glob_id_new if isinstance(r.place_glob_id_new, str) else None
            loc = settlement.get(glob) if glob else None
            stats["titles changed"] += r.title != r.title_new
            stats["places changed"] += (r.place_glob_id if isinstance(r.place_glob_id, str) else None) != glob
            if args.dry_run:
                continue
            conn.execute(text("UPDATE document SET title = :t, place_glob_id = :g, location_id = :l WHERE id = :d"),
                         {"t": f"[{r.title_new}]", "g": glob, "l": loc, "d": doc_id})
            if old_loc != loc:
                if old_loc:
                    conn.execute(text("DELETE FROM document_settlement WHERE document_id = :d AND settlement_id = :s"), {"d": doc_id, "s": old_loc})
                if loc:
                    conn.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) VALUES (:d, :s)"), {"d": doc_id, "s": loc})
                stats["settlements changed"] += 1
    for k, v in stats.items():
        print(f"{k}: {v:,}")
    if args.dry_run:
        print("dry run: nothing changed")


if __name__ == "__main__":
    main()
