#!/usr/bin/env python3
"""
Step 28: bring the place GLOB ids up to the places dataset of 2026-11-25
(data/places_v3_complete_20261125.xlsx) and the settlement ids the data team chose.

1. Java's Oostkust (GLOB_123) is merged into Java's (Noord-) Oostkust (GLOB2_1848):
   its documents (location_id, document_settlement) and its label move to that
   settlement, after the label shown; the settlement is removed.

2. Every settlement gets its current id (segmentation.doc_titles.Places.current):
   the chosen ids of SETTLEMENT_IDS (data/settlement-place-check.xlsx, column
   Choice; Sumatra's Westkust as its VOC area GLOB_434, not the island; Siam as
   the kingdom GLOB_539_HUM_2, not the general record GLOB_539), after
   the retired ids of deprecated_uris.

3. Every document.place_glob_id gets its current id the same way, which for the
   places of inferred titles also follows updated_uris: an id that became the
   general record of a place goes to the record with its old meaning (Colombo
   GLOB_806 -> the city GLOB_806_HUM_1).

4. Documents with a settlement but no place GLOB id get the settlement's id
   (Pontianak, GLOB3_115, was not in the places dataset of step 22), and inferred
   titles whose place is now a settlement's id get that settlement.

5. Index entries the OBP index gives two settlements ("Makassar; Banjarmasin",
   inventory 2163) are linked to both (document_settlement), with the segmentation
   documents that copied them; step 7 matched the whole value to one label only.
   Parts that are not a settlement (Ligor of "Siam;Ligor") are listed.

The names in inferred titles come from a new infer-titles run with step 24,
after this step. Running it again changes nothing.

  uv run python 28_update_place_ids.py [--dry-run]
"""

import argparse
import os
import zipfile

import pandas as pd
from sqlalchemy import create_engine, text

from segmentation.doc_titles import Places

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
OBP_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GLOBALISE - Digitized Indexes of the Dutch East India Company OBP (1602-1799).csv.zip")
CHOICES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "settlement-place-check.xlsx")
MERGE = ("GLOB_123", "GLOB2_1848")  # Java's Oostkust -> Java's (Noord-) Oostkust
NOT_THE_TEAMS = {"GLOB_434", "GLOB_539_HUM_2"}  # Sumatra's Westkust: the team chose the island (GLOB2_947), its VOC area was not on their list; Siam: decided later


def merge(conn, src_glob: str, dst_glob: str):
    ids = dict(conn.execute(text("SELECT glob_id, id FROM settlement WHERE glob_id IN (:a, :b)"), {"a": src_glob, "b": dst_glob}).fetchall())
    if src_glob not in ids:
        print(f"merge: no settlement {src_glob} (already merged)")
        return
    src, dst = ids[src_glob], ids[dst_glob]
    n = conn.execute(text("UPDATE document SET location_id = :d WHERE location_id = :s"), {"d": dst, "s": src}).rowcount
    conn.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) "
                      "SELECT document_id, :d FROM document_settlement WHERE settlement_id = :s"), {"d": dst, "s": src})
    m = conn.execute(text("DELETE FROM document_settlement WHERE settlement_id = :s"), {"s": src}).rowcount
    have = {r[0] for r in conn.execute(text("SELECT label FROM settlement_label WHERE settlement_id = :d"), {"d": dst})}
    for lid, label in conn.execute(text("SELECT id, label FROM settlement_label WHERE settlement_id = :s ORDER BY rowid"), {"s": src}).fetchall():
        conn.execute(text("DELETE FROM settlement_label WHERE id = :i"), {"i": lid})
        if label not in have:  # inserted again: after the label shown
            conn.execute(text("INSERT INTO settlement_label (id, label, settlement_id) VALUES (:i, :l, :d)"), {"i": lid, "l": label, "d": dst})
    conn.execute(text("DELETE FROM settlement WHERE id = :s"), {"s": src})
    print(f"merged settlement {src_glob} into {dst_glob}: {n:,} documents (location), {m:,} document_settlement rows")


def remap_settlements(conn, places: Places):
    rows = conn.execute(text("SELECT id, glob_id FROM settlement")).fetchall()
    used = {g for _, g in rows}
    for sid, glob in rows:
        new = places.current(glob)
        if new == glob:
            continue
        if new in used:
            raise SystemExit(f"settlement {glob} -> {new}: {new} is already another settlement's id")
        conn.execute(text("UPDATE settlement SET glob_id = :n WHERE id = :s"), {"n": new, "s": sid})
        used = (used - {glob}) | {new}
        print(f"settlement {glob} -> {new}")


def remap_documents(conn, places: Places):
    counts = conn.execute(text("SELECT place_glob_id, count(*) FROM document WHERE place_glob_id IS NOT NULL GROUP BY 1")).fetchall()
    total = 0
    for glob, n in sorted(counts, key=lambda r: -r[1]):
        new = places.current(glob)
        if new != glob:
            conn.execute(text("UPDATE document SET place_glob_id = :n WHERE place_glob_id = :o"), {"n": new, "o": glob})
            total += n
            print(f"  documents {glob} -> {new} ({places.label.get(new, '?')}): {n:,}")
    print(f"document place GLOB ids changed: {total:,}")


def fill_and_link(conn, places: Places):
    n = 0
    for sid, glob in conn.execute(text("SELECT id, glob_id FROM settlement")).fetchall():
        if glob in places.ids:
            n += conn.execute(text("UPDATE document SET place_glob_id = :g WHERE location_id = :s AND place_glob_id IS NULL"),
                              {"g": glob, "s": sid}).rowcount
    print(f"documents with a settlement that got its place GLOB id: {n:,}")
    docs = conn.execute(text(
        "SELECT d.id, s.id FROM document d JOIN settlement s ON s.glob_id = d.place_glob_id "
        "WHERE d.title_source = 'inferred' AND d.location_id IS NULL")).fetchall()
    for doc, sid in docs:
        conn.execute(text("UPDATE document SET location_id = :s WHERE id = :d"), {"s": sid, "d": doc})
        conn.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) VALUES (:d, :s)"), {"d": doc, "s": sid})
    print(f"inferred titles linked to the settlement of their place: {len(docs):,}")


def link_combined(conn):
    labels = {l.lower(): s for l, s in conn.execute(text("SELECT label, settlement_id FROM settlement_label"))}
    with zipfile.ZipFile(OBP_ZIP) as z:  # also holds a __MACOSX copy
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        obp = pd.read_csv(z.open(name), usecols=["ID", "SETTLEMENT"], dtype=str).dropna()
    obp = obp[obp.SETTLEMENT.str.contains(";")]
    added = 0
    for value, ids in obp.groupby("SETTLEMENT").ID:
        parts = [x.strip() for x in value.split(";") if x.strip()]
        missing = [x for x in parts if x.lower() not in labels]
        if missing:
            print(f"  {value!r}: not a settlement: {', '.join(missing)}")
        for oid in ids:
            docs = [r[0] for r in conn.execute(text(
                "SELECT de.document_id FROM document2external_id de JOIN external_id x ON x.id = de.external_id "
                "WHERE x.context = 'OBP_INDEX' AND x.identifier = :i"), {"i": oid})]
            for doc in docs:
                for part in parts:
                    if part.lower() in labels:
                        added += conn.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) VALUES (:d, :s)"),
                                              {"d": doc, "s": labels[part.lower()]}).rowcount
    print(f"settlements added for index entries with two settlements (and their copies): {added:,}")


def check(conn, places: Places):
    team = pd.read_excel(CHOICES)
    chosen = {str(c).split()[0] for c in team.iloc[:, 3]}
    globs = {g for (g,) in conn.execute(text("SELECT glob_id FROM settlement"))}
    print(f"settlements: {len(globs)}; not the team's choice: {sorted(globs - chosen)} (expected {sorted(NOT_THE_TEAMS)}); "
          f"team's choices not used: {sorted(chosen - globs - {MERGE[0], 'GLOB2_947', 'GLOB_539'})}")
    unknown = conn.execute(text("SELECT place_glob_id, count(*) FROM document WHERE place_glob_id IS NOT NULL GROUP BY 1")).fetchall()
    unknown = [(g, n) for g, n in unknown if g not in places.ids or places.current(g) != g]
    print(f"document place GLOB ids not current or not in the places dataset: {unknown or 'none'}")
    off = conn.execute(text("SELECT count(*) FROM document d JOIN settlement s ON s.id = d.location_id "
                            "WHERE d.place_glob_id IS NULL OR (d.place_glob_id != s.glob_id AND d.title_source IS NOT 'inferred')")).scalar()
    print(f"documents whose place GLOB id is not their settlement's: {off:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show the changes, then roll them back")
    args = ap.parse_args()
    places = Places()
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        tx = conn.begin()
        merge(conn, *MERGE)
        remap_settlements(conn, places)
        remap_documents(conn, places)
        fill_and_link(conn, places)
        link_combined(conn)
        check(conn, places)
        if args.dry_run:
            tx.rollback()
            print("dry run: nothing changed")
        else:
            tx.commit()


if __name__ == "__main__":
    main()
