#!/usr/bin/env python3
"""
Step 27: fix the settlement labels and GLOB ids.

1. 'Ban' is Bantam, not Banda. The OBP index gives 'Ban' as the settlement of
   20 entries in inventory 2081 (OBP ids 213714-213733), papers from Bantam of
   1728, and location_index.csv mapped that label to Banda (GLOB2_E_17). The
   label is removed (it is a truncated 'Bantam', not a spelling of a place),
   and the documents of inventory 2081 linked to Banda (those 20 entries and
   the 20 segmentation documents that copied them) are linked to Bantam
   instead: location_id, document_settlement and place_glob_id.

2. The app and the exports show a settlement by its first label (in the order
   the labels were inserted), which was 'Goede Hoop' for the Cape and 'Bengale'
   for Bengal. For settlements with several labels, the VOC name in PREFERRED
   is made the first label by inserting the other labels again after it.

3. Retired GLOB ids are replaced in settlement.glob_id and document.place_glob_id:
   the deprecated_uris of the places dataset (data/places_v3_complete_20261116.xlsx),
   and the Pontianak placeholder (EXTRA_ID_MAP, from the GLOBALISE data team).

Running it again changes nothing.

  uv run python 27_fix_settlements.py [--dry-run]
"""

import argparse
import os

import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")

BAN_INVENTORY = "2081"

# the label shown for settlements with more than one label
PREFERRED = {
    "Batavia", "Kaap de Goede Hoop", "Bengalen", "Makassar", "Malabar", "Banda", "Bantam",
    "Palembang", "Perzië", "Canton", "Siam", "Banjarmasin",
}

EXTRA_ID_MAP = {"GLOB_PONTIANAK_PLACEHOLDER": "GLOB3_115"}  # not in deprecated_uris


def glob_id_map() -> dict[str, str]:
    from segmentation.doc_titles import PLACES
    dep = pd.read_excel(PLACES, sheet_name="deprecated_uris")
    return {**dict(zip(dep.Old_ID.astype(str), dep.New_ID.astype(str))), **EXTRA_ID_MAP}


def settlement_with_label(conn, label: str):
    rows = conn.execute(text(
        "SELECT s.id, s.glob_id FROM settlement s JOIN settlement_label l ON l.settlement_id = s.id WHERE l.label = :l"),
        {"l": label}).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"expected one settlement with label {label!r}, found {len(rows)}")
    return rows[0]


def fix_ban(conn):
    banda, banda_glob = settlement_with_label(conn, "Banda")
    bantam, bantam_glob = settlement_with_label(conn, "Bantam")
    n = conn.execute(text("DELETE FROM settlement_label WHERE label = 'Ban'")).rowcount
    print(f"label 'Ban' removed: {n}")
    docs = [r[0] for r in conn.execute(text(
        """SELECT DISTINCT d.id FROM document d JOIN inventory i ON i.id = d.inventory_id
           LEFT JOIN document_settlement ds ON ds.document_id = d.id
           WHERE i.inventory_number = :inv AND (d.location_id = :b OR ds.settlement_id = :b)"""),
        {"inv": BAN_INVENTORY, "b": banda})]
    for doc in docs:
        conn.execute(text("UPDATE document SET location_id = :t WHERE id = :d AND location_id = :f"),
                     {"t": bantam, "f": banda, "d": doc})
        conn.execute(text("UPDATE document SET place_glob_id = :t WHERE id = :d AND place_glob_id = :f"),
                     {"t": bantam_glob, "f": banda_glob, "d": doc})
        conn.execute(text("DELETE FROM document_settlement WHERE document_id = :d AND settlement_id = :f"),
                     {"f": banda, "d": doc})
        conn.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) VALUES (:d, :t)"),
                     {"t": bantam, "d": doc})
    print(f"documents of inventory {BAN_INVENTORY} relinked from Banda to Bantam: {len(docs)}")


def order_labels(conn):
    labels = {}
    for row in conn.execute(text("SELECT id, settlement_id, label FROM settlement_label ORDER BY rowid")):
        labels.setdefault(row.settlement_id, []).append(row)
    glob = dict(conn.execute(text("SELECT id, glob_id FROM settlement")).fetchall())
    for sid, own in labels.items():
        if len(own) < 2:
            continue
        pref = [r for r in own if r.label in PREFERRED]
        if len(pref) != 1:
            print(f"no preferred label for {glob[sid]}: {' | '.join(r.label for r in own)}")
            continue
        if own[0].label == pref[0].label:
            continue
        for r in own:  # the others again, after the preferred label
            if r.label != pref[0].label:
                conn.execute(text("DELETE FROM settlement_label WHERE id = :i"), {"i": r.id})
                conn.execute(text("INSERT INTO settlement_label (id, label, settlement_id) VALUES (:i, :l, :s)"),
                             {"i": r.id, "l": r.label, "s": sid})
        print(f"{glob[sid]}: {pref[0].label}  (first label was {own[0].label!r})")


def remap_glob_ids(conn):
    used = {r[0] for r in conn.execute(text("SELECT glob_id FROM settlement"))}
    for old, new in glob_id_map().items():
        if old in used:
            if new in used:
                print(f"not changed: settlement {old} -> {new}, {new} is already another settlement's id")
            else:
                conn.execute(text("UPDATE settlement SET glob_id = :n WHERE glob_id = :o"), {"n": new, "o": old})
                used = (used - {old}) | {new}
                print(f"settlement {old} -> {new}")
        n = conn.execute(text("UPDATE document SET place_glob_id = :n WHERE place_glob_id = :o"), {"n": new, "o": old}).rowcount
        if n:
            print(f"documents {old} -> {new}: {n:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show the changes, then roll them back")
    args = ap.parse_args()
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        tx = conn.begin()
        fix_ban(conn)
        order_labels(conn)
        remap_glob_ids(conn)
        if args.dry_run:
            tx.rollback()
            print("dry run: nothing changed")
        else:
            tx.commit()


if __name__ == "__main__":
    main()
