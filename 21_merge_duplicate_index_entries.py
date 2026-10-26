#!/usr/bin/env python3
"""
Step 21: merge index entries that are the same document listed once per settlement.

The Digitized Indexes contain documents that were exported once for every
settlement they were linked to (e.g. inventory 3558, OBP 95419 'Batavia' and
95420 'Japan': same description, folios and years). Such a group becomes one
document: the entry with the lowest OBP id (the master) keeps its row and gets

  - all settlements of the group (table document_settlement; location_id stays
    the master's own settlement),
  - the index ids of the others (OBP_INDEX, TANAP, DIGITIZED TYPOSCRIPTS) and
    the NT index numbers of all (ExternalID context NT),
  - their document types (document2type, document2documenttype),

and the others are deleted. Duplicates: TANAP Digitized Index documents in the
same inventory with the same description (letters and digits only, lower case),
start and end folio and dates.

Also fills document_settlement with the location_id of every document, so that
the settlements of a document can be read from one table.

Segmentation documents of the deleted entries are replaced when the inventory
is segmented and imported again (run + import).

  uv run python 21_merge_duplicate_index_entries.py [--dry-run]
"""

import argparse
import os
import re
import uuid

import pandas as pd
from sqlalchemy import create_engine, text

from models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
INDEX_METHOD = "TANAP Digitized Index"
MOVED_CONTEXTS = ("OBP_INDEX", "TANAP", "DIGITIZED TYPOSCRIPTS", "NT")


def normalise(title):
    return re.sub(r"[^a-z0-9]", "", str(title or "").lower())


def duplicate_groups(conn, method_id):
    docs = pd.read_sql(
        text(
            "SELECT d.id, d.inventory_id, d.title, d.folio_start, d.folio_end, d.date_earliest_begin, d.date_latest_end, "
            "d.location_id, d.nt_index, (SELECT min(CAST(e.identifier AS INTEGER)) FROM document2external_id x "
            "JOIN external_id e ON e.id = x.external_id AND e.context = 'OBP_INDEX' WHERE x.document_id = d.id) AS obp "
            "FROM document d WHERE d.method_id = :m"
        ),
        conn,
        params={"m": method_id},
    )
    docs["key"] = docs.title.map(normalise)
    key = ["inventory_id", "key", "folio_start", "folio_end", "date_earliest_begin", "date_latest_end"]
    docs[key] = docs[key].astype(str)
    docs = docs.sort_values(["obp", "id"])
    groups = [g for _, g in docs.groupby(key, sort=False) if len(g) > 1]
    return docs, groups


def nt_external_id(conn, nt, cache):
    if nt not in cache:
        row = conn.execute(text("SELECT id FROM external_id WHERE context = 'NT' AND identifier = :i"), {"i": str(nt)}).fetchone()
        if row is None:
            ext = str(uuid.uuid4())
            conn.execute(text('INSERT INTO external_id (id, "URL", identifier, context) VALUES (:id, NULL, :i, \'NT\')'), {"id": ext, "i": str(nt)})
            cache[nt] = ext
        else:
            cache[nt] = row[0]
    return cache[nt]


def merge(conn, group, nt_cache):
    master, dups = group.iloc[0], group.iloc[1:]
    mid = master.id
    linked = {r[0] for r in conn.execute(text("SELECT external_id FROM document2external_id WHERE document_id = :d"), {"d": mid})}
    for nt in [master.nt_index, *dups.nt_index]:  # the NT index numbers of all entries, as ids of the master
        if pd.notna(nt):
            ext = nt_external_id(conn, int(nt), nt_cache)
            if ext not in linked:
                conn.execute(text("INSERT INTO document2external_id (id, document_id, external_id) VALUES (:id, :d, :e)"), {"id": str(uuid.uuid4()), "d": mid, "e": ext})
                linked.add(ext)
    settlements = {master.location_id} | set(dups.location_id)
    for s in settlements - {None, "None", "nan"}:
        conn.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) VALUES (:d, :s)"), {"d": mid, "s": s})
    for dup in dups.id:
        for ext_link, ext in conn.execute(text("SELECT x.id, x.external_id FROM document2external_id x WHERE x.document_id = :d"), {"d": dup}).fetchall():
            if ext in linked:
                conn.execute(text("DELETE FROM document2external_id WHERE id = :i"), {"i": ext_link})
            else:
                conn.execute(text("UPDATE document2external_id SET document_id = :m WHERE id = :i"), {"m": mid, "i": ext_link})
                linked.add(ext)
        for table, col in (("document2type", "document_type"), ("document2documenttype", "document_type_id")):
            have = {r[0] for r in conn.execute(text(f"SELECT {col} FROM {table} WHERE document_id = :d"), {"d": mid})}
            for rid, t in conn.execute(text(f"SELECT id, {col} FROM {table} WHERE document_id = :d"), {"d": dup}).fetchall():
                if t in have:
                    conn.execute(text(f"DELETE FROM {table} WHERE id = :i"), {"i": rid})
                else:
                    conn.execute(text(f"UPDATE {table} SET document_id = :m WHERE id = :i"), {"m": mid, "i": rid})
                    have.add(t)
        conn.execute(text("UPDATE document SET part_of_id = :m WHERE part_of_id = :d"), {"m": mid, "d": dup})
        for table in ("page2document", "document_evidence", "document_settlement"):
            conn.execute(text(f"DELETE FROM {table} WHERE document_id = :d"), {"d": dup})
        conn.execute(text("DELETE FROM document WHERE id = :d"), {"d": dup})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report the duplicate groups, change nothing")
    args = ap.parse_args()
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine, tables=[Base.metadata.tables["document_settlement"]])
    with engine.begin() as conn:
        method_id = conn.execute(text("SELECT id FROM document_identification_method WHERE name = :n"), {"n": INDEX_METHOD}).scalar()
        docs, groups = duplicate_groups(conn, method_id)
        n_dups = sum(len(g) - 1 for g in groups)
        print(f"index entries: {len(docs):,}; duplicate groups: {len(groups):,}; entries to merge into their master: {n_dups:,}")
        for g in groups[:3]:
            print(f"  master OBP {int(g.obp.iloc[0])}, merged: {[int(x) for x in g.obp.iloc[1:]]}  {str(g.title.iloc[0])[:80]}")
        if args.dry_run:
            return
        added = conn.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) "
                                  "SELECT id, location_id FROM document WHERE location_id IS NOT NULL")).rowcount
        print(f"document_settlement: {added:,} rows from location_id")
        nt_cache = {}
        for k, g in enumerate(groups, 1):
            merge(conn, g, nt_cache)
            if k % 200 == 0:
                print(f"  {k}/{len(groups)} groups merged")
        print(f"merged {n_dups:,} entries into {len(groups):,} masters")


if __name__ == "__main__":
    main()
