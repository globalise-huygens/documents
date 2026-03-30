"""
Import settlement data from location_index.csv into the database.

Reads data/location_index.csv, which maps textual settlement labels to
canonical GLOBALISE identifiers (e.g. 'GLOB2_894').  Multiple label rows
can share the same identifier (spelling variants, alternative names).

For each unique GLOB identifier this script creates one Settlement row.
Every label that maps to that identifier becomes a SettlementLabel row.

Run order: this is step 6 — run after scripts 1–5 and before script 7.

Output tables
─────────────
  settlement        one row per unique GLOB ID
  settlement_label  one row per (label, glob_id) pair

Upsert behaviour
────────────────
  • If a Settlement with the same glob_id already exists it is left unchanged.
  • If a SettlementLabel with the same (label, settlement_id) already exists
    it is left unchanged.
  • New settlements / labels that are not yet in the database are inserted.
"""

import os
import sys
import uuid
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models import Base, Settlement, SettlementLabel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CSV = SCRIPT_DIR / "data" / "location_index.csv"


# ── helpers ────────────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Read location_index.csv and return a clean DataFrame."""
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.where(pd.notnull(df), None)

    # Validate expected columns
    required = {"SETTLEMENT", "ID"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing expected columns in CSV: {missing}")
        sys.exit(1)

    # Strip whitespace
    df["SETTLEMENT"] = df["SETTLEMENT"].str.strip()
    df["ID"] = df["ID"].str.strip()

    # Drop rows with null label or ID
    before = len(df)
    df = df.dropna(subset=["SETTLEMENT", "ID"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null SETTLEMENT or ID.")

    logger.info(
        f"Loaded {len(df)} label rows from {csv_path.name} "
        f"({df['ID'].nunique()} unique GLOB IDs)."
    )
    return df


# ── core import logic ──────────────────────────────────────────────────────────

def import_settlements(csv_path: Path, database_url: str) -> dict:
    """
    Parse location_index.csv and upsert Settlement + SettlementLabel rows.

    Returns a stats dict: settlements_created, labels_created, skipped.
    """
    df = load_csv(csv_path)

    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)   # creates settlement / settlement_label if absent
    session = Session(engine)

    stats = {
        "settlements_created": 0,
        "labels_created": 0,
        "skipped_settlements": 0,
        "skipped_labels": 0,
    }

    try:
        # ── 1. Preload existing settlements keyed by glob_id ──────────────────
        existing_settlements: dict[str, str] = {}   # glob_id → settlement.id
        for row in session.execute(text("SELECT id, glob_id FROM settlement")).all():
            existing_settlements[row[1]] = row[0]

        # ── 2. Preload existing labels as (label, settlement_id) pairs ─────────
        existing_labels: set[tuple[str, str]] = set()
        for row in session.execute(
            text("SELECT label, settlement_id FROM settlement_label")
        ).all():
            existing_labels.add((row[0], row[1]))

        # ── 3. Build Settlement rows to insert ────────────────────────────────
        settlement_rows: list[dict] = []
        for glob_id in df["ID"].unique():
            if glob_id in existing_settlements:
                stats["skipped_settlements"] += 1
                continue
            new_id = str(uuid.uuid4())
            settlement_rows.append({"id": new_id, "glob_id": glob_id})
            existing_settlements[glob_id] = new_id   # make visible for label pass
            stats["settlements_created"] += 1

        if settlement_rows:
            session.execute(Settlement.__table__.insert(), settlement_rows)
            session.commit()
            logger.info(f"Inserted {len(settlement_rows)} new Settlement rows.")

        # ── 4. Build SettlementLabel rows to insert ───────────────────────────
        label_rows: list[dict] = []
        for _, csv_row in df.iterrows():
            label = csv_row["SETTLEMENT"]
            glob_id = csv_row["ID"]
            settlement_id = existing_settlements.get(glob_id)

            if settlement_id is None:
                # Should not happen after step 3, but guard anyway
                logger.warning(f"No settlement found for glob_id '{glob_id}' — skipping label '{label}'.")
                continue

            if (label, settlement_id) in existing_labels:
                stats["skipped_labels"] += 1
                continue

            label_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "label": label,
                    "settlement_id": settlement_id,
                }
            )
            existing_labels.add((label, settlement_id))   # guard intra-batch dupes
            stats["labels_created"] += 1

        if label_rows:
            session.execute(SettlementLabel.__table__.insert(), label_rows)
            session.commit()
            logger.info(f"Inserted {len(label_rows)} new SettlementLabel rows.")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return stats


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Import settlement data from location_index.csv into the database. "
            "Run this as step 6, after scripts 1–5 and before script 7."
        )
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Path to location_index.csv. "
            f"Defaults to data/location_index.csv relative to this script."
        ),
    )
    parser.add_argument(
        "--database",
        default=DATABASE_URL,
        help=f"SQLAlchemy database URL (default: {DATABASE_URL})",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV

    print("=" * 60)
    print("GLOBALISE — Settlement Import  (step 6 of 8)")
    print("=" * 60)
    print(f"Source : {csv_path}")
    print(f"DB     : {args.database}")
    print("=" * 60)

    stats = import_settlements(csv_path, args.database)

    print("\n=== Result ===")
    print(f"  Settlements created  : {stats['settlements_created']}")
    print(f"  Settlements skipped  : {stats['skipped_settlements']} (already existed)")
    print(f"  Labels created       : {stats['labels_created']}")
    print(f"  Labels skipped       : {stats['skipped_labels']} (already existed)")
    print("✓ Done.")


if __name__ == "__main__":
    main()
