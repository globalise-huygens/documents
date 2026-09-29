"""
Annotate OBP index documents with their place in the ToC structure.

Sources (see toc_sources.py): the OBP CSV, whose ID is the leading
identifier (ExternalID OBP_INDEX); the TANAP Excel (katern label, recto/verso
sides); and OBP NT_gecorrigeerd.xlsx (physical order, corrected katern
labels), matched to CSV ids by inventory, description and start folio.

Sets on every document linked to an OBP_INDEX id:
  - toc_katern            NT label (settlement + DEEL), else the TANAP Excel's
  - toc_deel              the CSV's SECTION (= DEEL)
  - toc_folio_start_side  Recto/Verso when the TANAP page range says so
  - toc_folio_end_side      (e.g. '14v-16')
  - nt_index              the entry's ID in NT_gecorrigeerd
  - toc_order             1-based position within the inventory. In inventories
                          covered by NT this is the physical NT order; entries NT
                          lacks get NULL. Elsewhere (e.g. typoscript inventories)
                          it follows the CSV id.
  - toc_folio_sequence    1-based foliation sequence (physical section): a new
                          sequence starts, in toc_order, where the start folio
                          drops to <= RESTART_MAX_FOLIO after the running
                          maximum reached >= RESTART_MIN_PREVIOUS. The katern
                          label is not used: in many volumes documents of
                          several settlements share one foliation.

Order and sequences are computed per inventory over the "TANAP Digitized
Index" documents, so run script 7.5 first. Safe to rerun.
"""

import argparse
import logging
import os
import time

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

import toc_sources
from models import Base

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
TANAP_METHOD_NAME = "TANAP Digitized Index"
RESTART_MAX_FOLIO = 3
RESTART_MIN_PREVIOUS = 10

NEW_COLUMNS = {
    "toc_katern": "TEXT",
    "toc_deel": "INTEGER",
    "toc_order": "INTEGER",
    "toc_folio_sequence": "INTEGER",
    "toc_folio_start_side": "VARCHAR(5)",
    "toc_folio_end_side": "VARCHAR(5)",
    "nt_index": "INTEGER",
}


def ensure_columns(engine):
    """Add the toc columns to document; renames the earlier toc_section column."""
    existing = {col["name"] for col in inspect(engine).get_columns("document")}
    with engine.begin() as conn:
        if "toc_section" in existing and "toc_katern" not in existing:
            conn.execute(text("ALTER TABLE document RENAME COLUMN toc_section TO toc_katern"))
            existing.add("toc_katern")
            logger.info("Renamed document.toc_section to toc_katern")
        for name, sql_type in NEW_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE document ADD COLUMN {name} {sql_type}"))
                logger.info("Added '%s' column to document table", name)


def folio_sequences(folio_starts) -> list[int]:
    """Number the foliation sequences of folio_starts (already in physical order)."""
    seq, running_max, out = 1, None, []
    for fs in folio_starts:
        if pd.notna(fs):
            if (
                running_max is not None
                and running_max >= RESTART_MIN_PREVIOUS
                and fs <= RESTART_MAX_FOLIO
                and fs < running_max
            ):
                seq += 1
                running_max = None
            running_max = fs if running_max is None else max(running_max, fs)
        out.append(seq)
    return out


def compute_entries(session: Session) -> pd.DataFrame:
    """One row per CSV id present in the database, with all toc_* values."""
    csv = toc_sources.load_obp_csv()
    nt = toc_sources.match_nt_to_csv(toc_sources.load_nt(), csv)
    tanap = toc_sources.load_tanap_excel()

    db = pd.DataFrame(
        session.execute(
            text(
                "SELECT CAST(e.identifier AS INTEGER), i.inventory_number "
                "FROM external_id e "
                "JOIN document2external_id de ON de.external_id = e.id "
                "JOIN document d ON d.id = de.document_id "
                "JOIN inventory i ON i.id = d.inventory_id "
                "JOIN document_identification_method m ON m.id = d.method_id "
                "WHERE e.context = 'OBP_INDEX' AND m.name = :name"
            ),
            {"name": TANAP_METHOD_NAME},
        ).all(),
        columns=["csv_id", "inventory"],
    ).drop_duplicates("csv_id")

    e = (
        db.merge(csv[["csv_id", "folio_start", "deel"]], on="csv_id", how="left")
        .merge(
            nt.dropna(subset=["csv_id"]).astype({"csv_id": int})[
                ["csv_id", "nt_index", "nt_katern"]
            ],
            on="csv_id",
            how="left",
        )
        .merge(tanap, on="csv_id", how="left")
    )

    nt_covered = e.groupby("inventory")["nt_index"].transform(lambda s: s.notna().any())
    e["order_key"] = e["nt_index"].where(nt_covered, e["csv_id"])
    e["toc_order"] = e.groupby("inventory")["order_key"].rank(method="first")
    e = e.sort_values(["inventory", "toc_order"])
    ordered = e["toc_order"].notna()
    e.loc[ordered, "toc_folio_sequence"] = (
        e[ordered]
        .groupby("inventory", group_keys=False)["folio_start"]
        .apply(lambda s: pd.Series(folio_sequences(s), index=s.index))
    )
    e["toc_katern"] = e["nt_katern"].fillna(e["tanap_katern"])

    logger.info(
        "%d entries in %d inventories: %d ordered by NT, %d by CSV id, %d unordered "
        "(NT inventory, entry missing from NT); %d inventories with >1 foliation sequence",
        len(e),
        e["inventory"].nunique(),
        (ordered & nt_covered).sum(),
        (ordered & ~nt_covered).sum(),
        (~ordered).sum(),
        (e.groupby("inventory")["toc_folio_sequence"].max() > 1).sum(),
    )
    return e


def main(db_url: str, dry_run: bool = False):
    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        entries = compute_entries(session)
        if dry_run:
            logger.info("Dry run — no columns added, nothing written")
            return

        Base.metadata.create_all(engine)
        ensure_columns(engine)
        t0 = time.time()

        def nullable_int(v):
            return None if pd.isna(v) else int(v)

        def nullable_str(v):
            return None if pd.isna(v) else v

        by_id = {
            r.csv_id: {
                "toc_katern": nullable_str(r.toc_katern),
                "toc_deel": nullable_int(r.deel),
                "toc_order": nullable_int(r.toc_order),
                "toc_folio_sequence": nullable_int(r.toc_folio_sequence),
                "toc_folio_start_side": nullable_str(r.folio_start_side),
                "toc_folio_end_side": nullable_str(r.folio_end_side),
                "nt_index": nullable_int(r.nt_index),
            }
            for r in entries.itertuples(index=False)
        }

        docs = session.execute(
            text(
                "SELECT CAST(e.identifier AS INTEGER), de.document_id "
                "FROM external_id e JOIN document2external_id de ON de.external_id = e.id "
                "WHERE e.context = 'OBP_INDEX'"
            )
        ).all()
        rows = [{"id": doc_id, **by_id[csv_id]} for csv_id, doc_id in docs if csv_id in by_id]

        session.execute(
            text(
                "UPDATE document SET toc_katern = NULL, toc_deel = NULL, toc_order = NULL, "
                "toc_folio_sequence = NULL, toc_folio_start_side = NULL, "
                "toc_folio_end_side = NULL, nt_index = NULL"
            )
        )
        session.execute(
            text(
                "UPDATE document SET toc_katern = :toc_katern, toc_deel = :toc_deel, "
                "toc_order = :toc_order, toc_folio_sequence = :toc_folio_sequence, "
                "toc_folio_start_side = :toc_folio_start_side, "
                "toc_folio_end_side = :toc_folio_end_side, nt_index = :nt_index "
                "WHERE id = :id"
            ),
            rows,
        )
        session.commit()
        logger.info(
            "Updated %d documents (%d OBP_INDEX links had no entry) in %.1fs",
            len(rows),
            len(docs) - len(rows),
            time.time() - t0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DATABASE_URL, help="Database URL")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = parser.parse_args()
    main(args.db, dry_run=args.dry_run)
